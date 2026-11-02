// =============================================================================
//  Stage C: geometric primitive landmarks   (GNSS-denied multi-terrain loc.)
// -----------------------------------------------------------------------------
//  Subscribes : /lio/odometry      (LIO IMU pose; drives keyframe gating)
//               /lio/cloud_registered (LIO scan, ALREADY in world frame)
//  Publishes  : /landmarks         (visualization_msgs/MarkerArray for RViz)
//
//  At keyframe rate (every ~keyframe_dist m or keyframe_time s) it extracts two
//  typed primitives from the accumulated world-frame cloud and fuses them into a
//  persistent constellation map (see landmark_map.hpp):
//
//    POLES  — vertical cylinders (trees/posts/pillars). Extracted by a
//             2D-FOOTPRINT clustering of a ground-relative height band: a sparse
//             32-beam trunk fragments under 3D Euclidean clustering (large ring
//             gaps), but its xy footprint stays compact, so we cluster columns in
//             2D and gate on height / footprint / aspect. Robust by design.
//    PLANES — vertical patches (walls/facades). Iterative RANSAC on the
//             non-ground points, keeping near-vertical planes with enough extent.
//
//  Scans are ACCUMULATED between keyframes (every /lio/cloud_registered is appended)
//  so trunks are dense enough to detect, then the buffer is consumed and cleared.
//
//  This stage builds + visualises the landmark map and lets us validate data
//  association; it does NOT correct the pose — that is the Stage-E backend, which
//  will consume these landmarks as factors. The LIO is untouched.
//
//  Algorithmic detail for the map (association/persistence/promotion) lives in:
//     include/copernicus_localization/landmark_map.hpp
// =============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include "copernicus_localization/landmark_map.hpp"
#include "copernicus_localization/msg/landmark_obs.hpp"

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;

class LandmarkNode : public rclcpp::Node
{
public:
  LandmarkNode() : Node("landmark_node")
  {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/lio/cloud_registered");
    odom_topic_  = declare_parameter<std::string>("odom_topic", "/lio/odometry");

    keyframe_dist_ = declare_parameter<double>("kf_dist", 1.0);
    keyframe_time_ = declare_parameter<double>("kf_time", 2.0);
    work_radius_   = declare_parameter<double>("work_radius", 30.0);
    max_accum_pts_ = declare_parameter<int>("max_accum_points", 400000);

    // ground model + height band (relative to a local min-z grid)
    ground_cell_ = declare_parameter<double>("ground_cell", 1.0);
    band_min_    = declare_parameter<double>("band_min", 0.5);
    band_max_    = declare_parameter<double>("band_max", 2.5);

    // pole extraction (2D footprint clustering)
    pole_cell_       = declare_parameter<double>("pole_cell", 0.30);
    pole_min_pts_    = declare_parameter<int>("pole_min_points", 8);
    pole_min_diam_   = declare_parameter<double>("pole_min_diameter", 0.05);
    pole_max_diam_   = declare_parameter<double>("pole_max_diameter", 1.0);
    pole_min_height_ = declare_parameter<double>("pole_min_height", 1.0);
    pole_max_foot_   = declare_parameter<double>("pole_max_footprint", 1.2);
    pole_aspect_     = declare_parameter<double>("pole_aspect", 1.5);  // height/footprint

    // plane extraction (RANSAC)
    enable_planes_   = declare_parameter<bool>("enable_planes", true);
    plane_min_h_     = declare_parameter<double>("plane_min_height", 0.3);
    plane_ransac_d_  = declare_parameter<double>("plane_ransac_dist", 0.10);
    plane_min_inl_   = declare_parameter<int>("plane_min_inliers", 100);
    // max RANSAC peels per keyframe: must exceed the count of dominant non-wall
    // planes (ground/canopy remnants) so vertical walls still get reached.
    plane_max_n_     = declare_parameter<int>("plane_max_count", 15);
    plane_vert_tol_  = declare_parameter<double>("plane_vertical_tol", 0.30); // |nz|<tol
    plane_min_ext_   = declare_parameter<double>("plane_min_extent", 1.2);

    cop::LandmarkConfig lc;
    lc.pole_assoc_dist  = declare_parameter<double>("assoc_pole_dist", 0.6);
    lc.plane_assoc_dist = declare_parameter<double>("assoc_plane_dist", 2.0);
    lc.plane_normal_tol = declare_parameter<double>("assoc_plane_normal_tol", 0.25);
    lc.plane_offset_tol = declare_parameter<double>("assoc_plane_offset_tol", 0.5);
    lc.promote_hits     = declare_parameter<int>("promote_hits", 3);
    lc.forget_frames    = declare_parameter<int>("forget_frames", 8);
    map_.configure(lc);

    // CONSUMER subscriptions are best-effort + shallow so this keyframe-rate node
    // can NEVER back-pressure the LIO front-end (a reliable subscriber that
    // falls behind throttles the publisher). Dropping scans we can't keep up with
    // is the correct behaviour here. Markers published reliable (low-rate output).
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::QoS(rclcpp::KeepLast(2)).best_effort(),
      std::bind(&LandmarkNode::cloudCb, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(rclcpp::KeepLast(20)).best_effort(),
      std::bind(&LandmarkNode::odomCb, this, std::placeholders::_1));
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/landmarks", rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
    // per-keyframe pole observations for the Stage-E backend (reliable: graph info)
    obs_pub_ = create_publisher<copernicus_localization::msg::LandmarkObs>(
      "/landmark_obs", rclcpp::QoS(rclcpp::KeepLast(50)).reliable());

    accum_.reset(new CloudT);
    RCLCPP_INFO(get_logger(),
      "landmark_node: %s + %s -> /landmarks | kf=%.1fm/%.1fs band=[%.1f,%.1f] planes=%s",
      cloud_topic_.c_str(), odom_topic_.c_str(), keyframe_dist_, keyframe_time_,
      band_min_, band_max_, enable_planes_ ? "on" : "off");
  }

private:
  void odomCb(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const auto & p = msg->pose.pose.position;
    odom_xyz_ = Eigen::Vector3f(p.x, p.y, p.z);
    have_odom_ = true;
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    if (!have_odom_) return;
    world_frame_ = msg->header.frame_id;
    CloudT scan;
    pcl::fromROSMsg(*msg, scan);
    // accumulate world-frame points within the working radius of the sensor
    const float r2 = float(work_radius_ * work_radius_);
    accum_->points.reserve(accum_->size() + scan.size());
    for (const auto & p : scan.points)
    {
      const float dx = p.x - odom_xyz_.x(), dy = p.y - odom_xyz_.y();
      if (dx * dx + dy * dy <= r2) accum_->points.push_back(p);
    }
    if (int(accum_->size()) > max_accum_pts_)        // safety bound if stationary
      accum_->points.erase(accum_->points.begin(),
                           accum_->points.begin() + (accum_->size() - max_accum_pts_));

    const rclcpp::Time now = msg->header.stamp;
    const bool first = !kf_init_;
    const double moved = first ? 1e9 : (odom_xyz_ - last_kf_xyz_).norm();
    const double dt    = first ? 1e9 : (now - last_kf_time_).seconds();
    if (!(first || moved >= keyframe_dist_ || dt >= keyframe_time_)) return;

    last_kf_xyz_ = odom_xyz_; last_kf_time_ = now; kf_init_ = true;
    ++frame_;
    cur_pole_obs_.clear();
    if (accum_->size() >= 50) extract();
    accum_->points.clear();
    publishMarkers(msg->header.stamp);
    publishObs(msg->header.stamp);
  }

  // Publish the poles observed THIS keyframe (world frame) + the keyframe position,
  // so the Stage-E backend can add pose<->landmark factors.
  void publishObs(const builtin_interfaces::msg::Time & stamp)
  {
    if (cur_pole_obs_.empty()) return;
    copernicus_localization::msg::LandmarkObs m;
    m.stamp = stamp;
    m.keyframe_position.x = odom_xyz_.x();
    m.keyframe_position.y = odom_xyz_.y();
    m.keyframe_position.z = odom_xyz_.z();
    for (const auto & p : cur_pole_obs_)
    {
      geometry_msgs::msg::Point q; q.x = p.x(); q.y = p.y(); q.z = p.z();
      m.positions.push_back(q);
    }
    obs_pub_->publish(m);
  }

  // ---- per-keyframe primitive extraction ---------------------------------
  void extract()
  {
    last_pts_ = accum_->size();                       // for the (post-clear) log
    const double inv_g = 1.0 / ground_cell_;
    // 1. local ground-height grid: min z per cell (+8-neighbourhood smoothing)
    std::unordered_map<int64_t, float> gmin;
    gmin.reserve(accum_->size());
    auto gkey = [inv_g](float x, float y) {
      int64_t ix = int64_t(std::floor(x * inv_g)), iy = int64_t(std::floor(y * inv_g));
      return (ix << 32) ^ (iy & 0xffffffff);
    };
    for (const auto & p : accum_->points)
    {
      const int64_t k = gkey(p.x, p.y);
      auto it = gmin.find(k);
      if (it == gmin.end() || p.z < it->second) gmin[k] = p.z;
    }
    auto ground_at = [&](float x, float y) -> float {
      const int64_t ix = int64_t(std::floor(x * inv_g)), iy = int64_t(std::floor(y * inv_g));
      float g = std::numeric_limits<float>::max();
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
        {
          auto it = gmin.find(((ix + dx) << 32) ^ ((iy + dy) & 0xffffffff));
          if (it != gmin.end()) g = std::min(g, it->second);
        }
      return g == std::numeric_limits<float>::max() ? 0.f : g;
    };

    // 2. split into height-band points (poles) and above-ground points (planes)
    CloudT::Ptr band(new CloudT), nonground(new CloudT);
    band->points.reserve(accum_->size());
    nonground->points.reserve(accum_->size());
    for (const auto & p : accum_->points)
    {
      const float h = p.z - ground_at(p.x, p.y);
      if (h >= band_min_ && h <= band_max_) band->points.push_back(p);
      if (h >= plane_min_h_) nonground->points.push_back(p);
    }

    last_poles_ = extractPoles(*band, ground_at);
    last_planes_ = enable_planes_ ? extractPlanes(nonground) : 0;
    map_.endFrame(frame_);
  }

  // POLES: 2D-footprint connected-component clustering of the height band.
  int extractPoles(const CloudT & band, const std::function<float(float, float)> & ground_at)
  {
    const double inv = 1.0 / pole_cell_;
    auto key = [inv](float x, float y) {
      int64_t ix = int64_t(std::floor(x * inv)), iy = int64_t(std::floor(y * inv));
      return (ix << 32) ^ (iy & 0xffffffff);
    };
    struct Cell { std::vector<int> idx; };
    std::unordered_map<int64_t, Cell> cells;
    cells.reserve(band.size());
    for (int i = 0; i < int(band.size()); ++i)
      cells[key(band.points[i].x, band.points[i].y)].idx.push_back(i);

    std::unordered_map<int64_t, char> visited;
    std::vector<int64_t> stack;
    int found = 0;
    for (const auto & kv : cells)
    {
      if (visited[kv.first]) continue;
      stack.clear(); std::vector<int> comp;
      stack.push_back(kv.first); visited[kv.first] = 1;
      while (!stack.empty())
      {
        const int64_t cur = stack.back(); stack.pop_back();
        auto it = cells.find(cur);
        if (it != cells.end()) comp.insert(comp.end(), it->second.idx.begin(), it->second.idx.end());
        const int64_t cx = cur >> 32;
        const int32_t cy = int32_t(cur & 0xffffffff);
        for (int dx = -1; dx <= 1; ++dx)
          for (int dy = -1; dy <= 1; ++dy)
          {
            if (!dx && !dy) continue;
            const int64_t nk = ((cx + dx) << 32) ^ ((cy + dy) & 0xffffffff);
            if (cells.count(nk) && !visited[nk]) { visited[nk] = 1; stack.push_back(nk); }
          }
      }
      if (int(comp.size()) < pole_min_pts_) continue;

      float xmin = 1e30f, xmax = -1e30f, ymin = 1e30f, ymax = -1e30f, zmin = 1e30f, zmax = -1e30f;
      double sx = 0, sy = 0;
      for (int id : comp)
      {
        const auto & p = band.points[id];
        xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z);
        sx += p.x; sy += p.y;
      }
      const float foot = std::max(xmax - xmin, ymax - ymin);
      const float diam = std::max(0.02f, foot);
      const float height = zmax - zmin;
      if (foot > pole_max_foot_ || diam < pole_min_diam_ || diam > pole_max_diam_) continue;
      if (height < pole_min_height_) continue;
      if (height < pole_aspect_ * foot) continue;          // tall & thin only
      const float cx = float(sx / comp.size()), cy = float(sy / comp.size());
      Eigen::Vector3f base(cx, cy, ground_at(cx, cy));
      map_.observePole(base, diam, height, frame_);
      cur_pole_obs_.push_back(base);                   // for the backend landmark factors
      ++found;
    }
    return found;
  }

  // PLANES: iterative RANSAC, keep near-vertical patches with enough extent.
  int extractPlanes(CloudT::Ptr cloud)
  {
    int found = 0;
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(plane_ransac_d_);
    pcl::ExtractIndices<PointT> ext;

    for (int iter = 0; iter < plane_max_n_ && int(cloud->size()) >= plane_min_inl_; ++iter)
    {
      pcl::ModelCoefficients coeff;
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      seg.setInputCloud(cloud);
      seg.segment(*inliers, coeff);
      if (int(inliers->indices.size()) < plane_min_inl_ || coeff.values.size() < 4) break;

      Eigen::Vector3f n(coeff.values[0], coeff.values[1], coeff.values[2]);
      n.normalize();
      // centroid + extent of the inlier patch
      Eigen::Vector3f c(0, 0, 0);
      Eigen::Vector3f mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
      for (int id : inliers->indices)
      {
        const auto & p = cloud->points[id];
        Eigen::Vector3f v(p.x, p.y, p.z);
        c += v; mn = mn.cwiseMin(v); mx = mx.cwiseMax(v);
      }
      c /= float(inliers->indices.size());
      const float extent = (mx - mn).norm();

      const bool vertical = std::fabs(n.z()) < plane_vert_tol_;
      if (vertical && extent >= plane_min_ext_)
      {
        map_.observePlane(c, n, extent, frame_);
        ++found;
      }
      // peel off this plane's inliers and continue searching
      ext.setInputCloud(cloud);
      ext.setIndices(inliers);
      ext.setNegative(true);
      CloudT::Ptr rest(new CloudT);
      ext.filter(*rest);
      cloud.swap(rest);
    }
    return found;
  }

  // ---- visualization ------------------------------------------------------
  void publishMarkers(const builtin_interfaces::msg::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.header.frame_id = world_frame_;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    int mid = 0;
    for (const auto & lm : map_.landmarks())
    {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = world_frame_;
      m.header.stamp = stamp;
      m.id = mid++;
      m.action = visualization_msgs::msg::Marker::ADD;
      const float a = lm.confirmed ? 0.85f : 0.30f;

      if (lm.type == cop::LandmarkType::Pole)
      {
        m.ns = "poles";
        m.type = visualization_msgs::msg::Marker::CYLINDER;
        m.pose.position.x = lm.pos.x();
        m.pose.position.y = lm.pos.y();
        m.pose.position.z = lm.pos.z() + lm.height * 0.5f;
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = std::max(0.05f, lm.diameter);
        m.scale.z = std::max(0.1f, lm.height);
        // green confirmed, yellow candidate; brighter = more durable
        m.color.r = lm.confirmed ? 0.1f : 0.9f;
        m.color.g = 0.9f; m.color.b = 0.1f; m.color.a = a;
      }
      else
      {
        m.ns = "planes";
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.pose.position.x = lm.pos.x();
        m.pose.position.y = lm.pos.y();
        m.pose.position.z = lm.pos.z();
        Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(
          Eigen::Vector3f::UnitZ(), lm.normal);
        m.pose.orientation.x = q.x(); m.pose.orientation.y = q.y();
        m.pose.orientation.z = q.z(); m.pose.orientation.w = q.w();
        m.scale.x = std::max(0.5f, lm.extent);
        m.scale.y = std::max(0.5f, lm.extent);
        m.scale.z = 0.1f;
        m.color.r = 0.2f; m.color.g = 0.4f; m.color.b = 1.0f; m.color.a = a;
      }
      arr.markers.push_back(m);
    }
    marker_pub_->publish(arr);
  }

  // params
  std::string cloud_topic_, odom_topic_;
  std::string world_frame_ = "odom";   // frame of the incoming LIO clouds
  double keyframe_dist_, keyframe_time_, work_radius_;
  int max_accum_pts_;
  double ground_cell_, band_min_, band_max_;
  double pole_cell_, pole_min_diam_, pole_max_diam_, pole_min_height_, pole_max_foot_, pole_aspect_;
  int pole_min_pts_;
  bool enable_planes_;
  double plane_min_h_, plane_ransac_d_, plane_vert_tol_, plane_min_ext_;
  int plane_min_inl_, plane_max_n_;

  // state
  cop::LandmarkMap map_;
  CloudT::Ptr accum_;
  Eigen::Vector3f odom_xyz_{0, 0, 0}, last_kf_xyz_{0, 0, 0};
  bool have_odom_ = false, kf_init_ = false;
  rclcpp::Time last_kf_time_{0, 0, RCL_ROS_TIME};
  uint32_t frame_ = 0;
  int last_poles_ = 0, last_planes_ = 0;
  size_t last_pts_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::vector<Eigen::Vector3f> cur_pole_obs_;            // poles observed this keyframe (world)
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<copernicus_localization::msg::LandmarkObs>::SharedPtr obs_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LandmarkNode>());
  rclcpp::shutdown();
  return 0;
}
