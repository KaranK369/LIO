// =============================================================================
//  §7  Environment router — scene classification + degeneracy (NO camera)
// -----------------------------------------------------------------------------
//  Subscribes : /lio/localizability (Localizability, per scan — PRIMARY)
//               /lio/cloud_registered (LIO odom-frame scan, accumulated)
//               /Odometry          (keyframe gating + crop centre)
//  Publishes  : /environment/state (copernicus_localization/EnvironmentState)
//               /environment/marker (RViz text: tier + degeneracy)
//
//  Every keyframe the router emits a tier (RICH / SPARSE / POOR) plus per-axis
//  localizability, which the rest of the stack routes on: the backend inflates the
//  LIO between-factor along weak axes, the preprocessor stops deleting structure,
//  the loop detector can stand down.
//
//  1. LOCALIZABILITY (primary, 6-DOF) — straight from the LIO's measurement
//     Jacobian: the normalised point-to-plane information of the points the iEKF
//     ACTUALLY used (Zhang-2016). Averaged over `loc_window_sec`, eigen-analysed in
//     translation and rotation. This sees what a cloud heuristic cannot: z/roll/
//     pitch/YAW degeneracy, the effect of the map, and the exact correspondence set.
//     Fallback when /lio/localizability is absent: horizontal wall/pole information
//     from voxel-PCA of the accumulated cloud (the original §7 estimate).
//  2. DIMENSIONALITY (scene mix, reporting) — voxel PCA (3x3x3 neighbourhood
//     moments) over a cloud ACCUMULATED across the keyframe interval. A single
//     sparse scan (VLP-16) puts one ring arc in a voxel neighbourhood, which reads
//     as a "horizontal line" everywhere and biases the mix toward scatter.
//  3. HYSTERESIS — degrade fast (enter_kf), recover slowly (exit_kf); no flicker.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "copernicus_localization/msg/localizability.hpp"

#include "copernicus_localization/msg/environment_state.hpp"

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using EnvMsg = copernicus_localization::msg::EnvironmentState;
using LocMsg = copernicus_localization::msg::Localizability;

class EnvironmentRouterNode : public rclcpp::Node
{
public:
  EnvironmentRouterNode() : Node("environment_router_node")
  {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/lio/cloud_registered");
    odom_topic_  = declare_parameter<std::string>("odom_topic", "/lio/odometry");
    loc_topic_   = declare_parameter<std::string>("localizability_topic", "/lio/localizability");
    kf_dist_     = declare_parameter<double>("kf_dist", 1.0);
    kf_time_     = declare_parameter<double>("kf_time", 2.0);
    work_radius_ = declare_parameter<double>("work_radius", 30.0);
    max_accum_pts_ = declare_parameter<int>("max_accum_points", 400000);

    // voxel-PCA dimensionality
    pca_cell_    = declare_parameter<double>("pca_cell", 0.40);     // voxel size [m]
    min_voxel_pts_ = declare_parameter<int>("min_voxel_pts", 6);    // 3x3x3 floor for PCA
    linear_thresh_  = declare_parameter<double>("linearity_thresh", 0.50);
    planar_thresh_  = declare_parameter<double>("planarity_thresh", 0.50);
    wall_nz_max_ = declare_parameter<double>("wall_normal_z_max", 0.50);
    pole_nz_min_ = declare_parameter<double>("pole_dir_z_min", 0.70);

    // LIO localizability (normalised eigenvalues: translation sums to 1)
    loc_window_   = declare_parameter<double>("loc_window_sec", 1.0);
    loc_ref_t_    = declare_parameter<double>("loc_ref_trans", 0.10);   // >= this = fully constrained
    loc_ref_r_    = declare_parameter<double>("loc_ref_rot", 0.10);
    poor_trans_   = declare_parameter<double>("poor_trans_eig", 0.01);  // = LIO remap thresh
    poor_rot_     = declare_parameter<double>("poor_rot_eig", 0.01);
    sparse_trans_ = declare_parameter<double>("sparse_trans_eig", 0.04);
    sparse_rot_   = declare_parameter<double>("sparse_rot_eig", 0.04);

    // cloud fallback thresholds (§7, horizontal-only)
    min_struct_voxels_ = declare_parameter<int>("min_struct_voxels", 25);
    min_localizability_ = declare_parameter<double>("min_localizability", 8.0);
    poor_deg_   = declare_parameter<double>("poor_degeneracy", 0.80);
    rich_deg_   = declare_parameter<double>("rich_degeneracy", 0.50);

    enter_kf_ = declare_parameter<int>("tier_enter_kf", 1);   // keyframes to adopt a WORSE tier
    exit_kf_  = declare_parameter<int>("tier_exit_kf", 3);    // keyframes to adopt a BETTER tier

    accum_.reset(new CloudT);
    const auto sub_qos = rclcpp::SensorDataQoS();
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, sub_qos,
      std::bind(&EnvironmentRouterNode::cloudCb, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
      std::bind(&EnvironmentRouterNode::odomCb, this, std::placeholders::_1));
    loc_sub_ = create_subscription<LocMsg>(
      loc_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
      [this](const LocMsg::ConstSharedPtr m) {
        loc_buf_.push_back(m);
        while (loc_buf_.size() > 200) loc_buf_.pop_front();
      });
    state_pub_ = create_publisher<EnvMsg>(
      "/environment/state", rclcpp::QoS(rclcpp::KeepLast(20)).reliable());
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      "/environment/marker", rclcpp::QoS(rclcpp::KeepLast(5)).reliable());

    RCLCPP_INFO(get_logger(),
      "environment_router: %s (primary) + %s -> /environment/state | kf=%.1fm/%.1fs | "
      "POOR<%.3f SPARSE<%.3f (normalised eig)",
      loc_topic_.c_str(), cloud_topic_.c_str(), kf_dist_, kf_time_, poor_trans_, sparse_trans_);
  }

private:
  static const char * tierName(uint8_t c)
  { return c == EnvMsg::RICH ? "RICH" : (c == EnvMsg::SPARSE ? "SPARSE" : "POOR"); }

  void odomCb(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    odom_xyz_ = Eigen::Vector3d(
      msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    have_odom_ = true;
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    if (!have_odom_) return;
    world_frame_ = msg->header.frame_id;
    CloudT cloud;
    pcl::fromROSMsg(*msg, cloud);
    const double r2 = work_radius_ * work_radius_;
    for (const auto & p : cloud.points)
    {
      if (int(accum_->size()) >= max_accum_pts_) break;
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      const double dx = p.x - odom_xyz_.x(), dy = p.y - odom_xyz_.y();
      if (dx * dx + dy * dy <= r2) accum_->points.push_back(p);
    }

    const rclcpp::Time now(msg->header.stamp);
    const bool first = !kf_init_;
    const double moved = (odom_xyz_ - last_kf_xyz_).norm();
    const double dt = kf_init_ ? (now - last_kf_time_).seconds() : 1e9;
    if (!(first || moved >= kf_dist_ || dt >= kf_time_)) return;
    last_kf_xyz_ = odom_xyz_; last_kf_time_ = now; kf_init_ = true;

    process(msg->header.stamp);
    accum_->points.clear();
  }

  static int64_t packk(int64_t ix, int64_t iy, int64_t iz)
  { return ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF); }

  // Voxel-PCA over the accumulated cloud: dimensionality mix + horizontal
  // wall/pole information (the cloud fallback for localizability).
  void dimensionality(int & n_plane, int & n_pole, int & n_scatter, int & n_horiz, Eigen::Matrix2d & Hxy) const
  {
    struct Mom { int n=0; double sx=0,sy=0,sz=0, xx=0,xy=0,xz=0,yy=0,yz=0,zz=0; };
    const double inv = 1.0 / pca_cell_;
    std::unordered_map<int64_t, Mom> vox;
    vox.reserve(accum_->size());
    for (const auto & p : accum_->points)
    {
      Mom & m = vox[packk(int64_t(std::floor(p.x * inv)), int64_t(std::floor(p.y * inv)), int64_t(std::floor(p.z * inv)))];
      m.n++;
      m.sx += p.x; m.sy += p.y; m.sz += p.z;
      m.xx += double(p.x)*p.x; m.xy += double(p.x)*p.y; m.xz += double(p.x)*p.z;
      m.yy += double(p.y)*p.y; m.yz += double(p.y)*p.z; m.zz += double(p.z)*p.z;
    }

    Hxy.setZero();
    n_plane = n_pole = n_scatter = n_horiz = 0;
    auto sx21 = [](int64_t v){ return (v & 0x100000) ? (v | ~int64_t(0x1FFFFF)) : v; };
    for (const auto & kv : vox)
    {
      const int64_t key = kv.first;
      const int64_t cx = sx21((key >> 42) & 0x1FFFFF), cy = sx21((key >> 21) & 0x1FFFFF), cz = sx21(key & 0x1FFFFF);
      Mom a;
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz)
          {
            auto it = vox.find(packk(cx+dx, cy+dy, cz+dz));
            if (it == vox.end()) continue;
            const Mom & m = it->second;
            a.n += m.n; a.sx += m.sx; a.sy += m.sy; a.sz += m.sz;
            a.xx += m.xx; a.xy += m.xy; a.xz += m.xz;
            a.yy += m.yy; a.yz += m.yz; a.zz += m.zz;
          }
      if (a.n < min_voxel_pts_) continue;

      const double N = a.n, mx = a.sx/N, my = a.sy/N, mz = a.sz/N;
      Eigen::Matrix3d cov;
      cov(0,0) = a.xx/N - mx*mx; cov(0,1) = a.xy/N - mx*my; cov(0,2) = a.xz/N - mx*mz;
      cov(1,1) = a.yy/N - my*my; cov(1,2) = a.yz/N - my*mz; cov(2,2) = a.zz/N - mz*mz;
      cov(1,0) = cov(0,1); cov(2,0) = cov(0,2); cov(2,1) = cov(1,2);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
      const Eigen::Vector3d ev = es.eigenvalues();
      const double e2 = std::max(ev[2], 1e-9), e1 = std::max(ev[1], 0.0), e0 = std::max(ev[0], 0.0);
      const double linearity = (e2 - e1) / e2;
      const double planarity = (e1 - e0) / e2;

      if (planarity >= planar_thresh_ && planarity >= linearity)
      {
        const Eigen::Vector3d n = es.eigenvectors().col(0);
        ++n_plane;
        if (std::abs(n.z()) < wall_nz_max_)
        {
          Eigen::Vector2d nxy(n.x(), n.y());
          const double nn = nxy.norm();
          if (nn > 1e-6) { nxy /= nn; Hxy += planarity * (nxy * nxy.transpose()); ++n_horiz; }
        }
      }
      else if (linearity >= linear_thresh_)
      {
        const Eigen::Vector3d d = es.eigenvectors().col(2);
        if (std::abs(d.z()) > pole_nz_min_) { Hxy += linearity * Eigen::Matrix2d::Identity(); ++n_pole; ++n_horiz; }
        else ++n_scatter;
      }
      else ++n_scatter;
    }
  }

  // Average the LIO's normalised information over the window ending at `stamp`.
  bool lioInfo(const rclcpp::Time & stamp, Eigen::Matrix3d & It, Eigen::Matrix3d & Ir) const
  {
    It.setZero(); Ir.setZero();
    int n = 0;
    for (const auto & m : loc_buf_)
    {
      const double age = (stamp - rclcpp::Time(m->header.stamp)).seconds();
      if (age < -0.2 || age > loc_window_ || m->num_effective < 6) continue;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) { It(r, c) += m->trans_info[r * 3 + c]; Ir(r, c) += m->rot_info[r * 3 + c]; }
      ++n;
    }
    if (n == 0) return false;
    It /= n; Ir /= n;
    return true;
  }

  void process(const builtin_interfaces::msg::Time & stamp)
  {
    int n_plane, n_pole, n_scatter, n_horiz;
    Eigen::Matrix2d Hxy;
    dimensionality(n_plane, n_pole, n_scatter, n_horiz, Hxy);
    const int n_total = n_plane + n_pole + n_scatter;

    EnvMsg out;
    out.header.stamp = stamp;
    out.header.frame_id = world_frame_;
    out.num_voxels = uint32_t(n_horiz);
    out.plane_ratio   = n_total ? float(n_plane) / n_total : 0.f;
    out.pole_ratio    = n_total ? float(n_pole) / n_total : 0.f;
    out.scatter_ratio = n_total ? float(n_scatter) / n_total : 0.f;

    Eigen::Vector3d loc_t(0, 0, 1), loc_r(1, 1, 1);
    Eigen::Matrix3d ax_t = Eigen::Matrix3d::Identity(), ax_r = Eigen::Matrix3d::Identity();
    double degeneracy;
    uint8_t raw;
    Eigen::Matrix3d It, Ir;
    if (lioInfo(rclcpp::Time(stamp), It, Ir))
    {
      // ---- PRIMARY: the LIO measurement-Jacobian localizability (6-DOF)
      out.source = EnvMsg::SOURCE_LIO;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> est(It), esr(Ir);
      const Eigen::Vector3d lt = est.eigenvalues(), lr = esr.eigenvalues();   // ascending
      ax_t = est.eigenvectors(); ax_r = esr.eigenvectors();
      for (int k = 0; k < 3; ++k)
      {
        loc_t[k] = std::clamp(lt[k] / loc_ref_t_, 0.0, 1.0);
        loc_r[k] = std::clamp(lr[k] / loc_ref_r_, 0.0, 1.0);
      }
      degeneracy = 1.0 - std::min(loc_t[0], loc_r[0]);
      if (lt[0] < poor_trans_ || lr[0] < poor_rot_)            raw = EnvMsg::POOR;
      else if (lt[0] < sparse_trans_ || lr[0] < sparse_rot_)   raw = EnvMsg::SPARSE;
      else                                                     raw = EnvMsg::RICH;
    }
    else
    {
      // ---- FALLBACK: horizontal wall/pole information from the accumulated cloud
      out.source = EnvMsg::SOURCE_CLOUD;
      if (n_horiz < min_struct_voxels_ || Hxy.trace() < min_localizability_)
      {
        degeneracy = 1.0;
        loc_t[0] = loc_t[1] = 0.0;
      }
      else
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> esH(Hxy);
        const Eigen::Vector2d lam = esH.eigenvalues();
        const Eigen::Matrix2d V = esH.eigenvectors();
        loc_t[0] = std::clamp(lam[0] / std::max(lam[1], 1e-9), 0.0, 1.0);
        loc_t[1] = 1.0;
        ax_t.col(0) = Eigen::Vector3d(V(0, 0), V(1, 0), 0.0);
        ax_t.col(1) = Eigen::Vector3d(V(0, 1), V(1, 1), 0.0);
        ax_t.col(2) = Eigen::Vector3d(0, 0, 1);
        degeneracy = 1.0 - loc_t[0];
      }
      if (n_horiz < min_struct_voxels_ || Hxy.trace() < min_localizability_ || degeneracy >= poor_deg_)
        raw = EnvMsg::POOR;
      else if (degeneracy <= rich_deg_) raw = EnvMsg::RICH;
      else raw = EnvMsg::SPARSE;
    }
    out.degeneracy = float(degeneracy);
    out.environment_class = hysteresis(raw);

    for (int k = 0; k < 3; ++k)
    {
      geometry_msgs::msg::Vector3 a, b;
      a.x = ax_t(0, k); a.y = ax_t(1, k); a.z = ax_t(2, k);
      b.x = ax_r(0, k); b.y = ax_r(1, k); b.z = ax_r(2, k);
      out.loc_axes.push_back(a);
      out.loc_eigvals.push_back(float(loc_t[k]));
      out.rot_axes.push_back(b);
      out.rot_eigvals.push_back(float(loc_r[k]));
    }

    state_pub_->publish(out);
    publishMarker(out);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "router[%s]: %-6s deg=%.2f | loc_t=[%.2f %.2f %.2f] loc_r=[%.2f %.2f %.2f] | plane=%.2f pole=%.2f scatter=%.2f",
      out.source == EnvMsg::SOURCE_LIO ? "lio" : "cloud",
      tierName(out.environment_class), out.degeneracy,
      loc_t[0], loc_t[1], loc_t[2], loc_r[0], loc_r[1], loc_r[2],
      out.plane_ratio, out.pole_ratio, out.scatter_ratio);
  }

  // degrade after enter_kf consecutive worse keyframes, recover after exit_kf better
  uint8_t hysteresis(uint8_t raw)
  {
    if (raw == tier_) { pending_count_ = 0; return tier_; }
    if (raw != pending_tier_) { pending_tier_ = raw; pending_count_ = 0; }
    ++pending_count_;
    const int need = raw > tier_ ? enter_kf_ : exit_kf_;     // larger enum value = worse
    if (pending_count_ >= need) { tier_ = raw; pending_count_ = 0; }
    return tier_;
  }

  void publishMarker(const EnvMsg & s)
  {
    visualization_msgs::msg::Marker m;
    m.header = s.header;
    m.ns = "environment"; m.id = 0;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = odom_xyz_.x();
    m.pose.position.y = odom_xyz_.y();
    m.pose.position.z = odom_xyz_.z() + 3.0;
    m.pose.orientation.w = 1.0;
    m.scale.z = 1.2;
    m.color.a = 1.0;
    if (s.environment_class == EnvMsg::RICH)        { m.color.r = 0.1f; m.color.g = 0.9f; m.color.b = 0.1f; }
    else if (s.environment_class == EnvMsg::SPARSE) { m.color.r = 0.9f; m.color.g = 0.9f; m.color.b = 0.1f; }
    else                                            { m.color.r = 0.9f; m.color.g = 0.1f; m.color.b = 0.1f; }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s  deg=%.2f", tierName(s.environment_class), s.degeneracy);
    m.text = buf;
    marker_pub_->publish(m);
  }

  // params
  std::string cloud_topic_, odom_topic_, loc_topic_;
  std::string world_frame_ = "odom";   // frame of the incoming LIO clouds
  double kf_dist_, kf_time_, work_radius_, pca_cell_;
  int max_accum_pts_, min_voxel_pts_, min_struct_voxels_;
  double linear_thresh_, planar_thresh_, wall_nz_max_, pole_nz_min_;
  double loc_window_, loc_ref_t_, loc_ref_r_, poor_trans_, poor_rot_, sparse_trans_, sparse_rot_;
  double min_localizability_, poor_deg_, rich_deg_;
  int enter_kf_, exit_kf_;

  // state
  CloudT::Ptr accum_;
  std::deque<LocMsg::ConstSharedPtr> loc_buf_;
  Eigen::Vector3d odom_xyz_{0, 0, 0}, last_kf_xyz_{0, 0, 0};
  bool have_odom_ = false, kf_init_ = false;
  rclcpp::Time last_kf_time_;
  uint8_t tier_ = EnvMsg::RICH, pending_tier_ = EnvMsg::RICH;
  int pending_count_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<LocMsg>::SharedPtr loc_sub_;
  rclcpp::Publisher<EnvMsg>::SharedPtr state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EnvironmentRouterNode>());
  rclcpp::shutdown();
  return 0;
}
