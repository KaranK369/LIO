// =============================================================================
//  Stage A: LiDAR point-cloud preprocessing  (GNSS-denied forest localization)
// -----------------------------------------------------------------------------
//  Subscribes  : /ouster/points   (Ouster OS1-32: x y z intensity t reflectivity ring ambient range)
//                /lio/odometry     (LIO IMU pose, only for dynamic removal)
//  Publishes   : /ouster/points_processed   (same Ouster format, cleaned)
//
//  Implements the full methodology Stage A, in the order the data flows:
//
//    0. TIME FIX (always on, REQUIRED) ----------------------------------------
//       Ouster per-point `t` is a uint32 nanosecond offset from scan start
//       (always non-negative).  We convert to float seconds internally, normalise
//       to start from 0, and convert back to uint32_t ns for output.
//       the LIO's oust64_handler reads the `t` field with timestamp_unit: 3 (ns).
//
//    0b. Particulate denoise --- DROR range-adaptive outlier removal: strips
//       blowing snow/dust/sand & rain (sparse isolated returns) BEFORE anything
//       else sees them. Distinct from foliage rejection (5) and dynamics (6).
//    1. Range crop ------------- drop ego/mount returns and far-sparse points.
//    2. Multi-return / intensity ---- drop weak returns (foliage/dust); if the
//       driver exposes a dual-return/echo field, keep only the strong echo.
//    3. Ground segmentation ---- Patchwork++-style (CZM + region-wise plane fit
//       + uprightness/flatness/elevation gating): rigid ground vs non-ground.
//    4. Lower-trunk band ------- keep ground + non-ground within [band_min,
//       band_max] ABOVE the *fitted* ground (slope-aware). Drops swaying canopy;
//       lower trunks barely move in wind. Primary wind defense, nearly free.
//    5. Rigidity classification - local structure tensor on the non-ground band:
//       keep linear (trunks) / planar, drop scattered/volumetric (foliage).
//    6. Dynamic-object removal -- Dynablox-style temporal free-space check at
//       keyframe rate: remove *translating* actors (needs odometry). In-place
//       wind jitter is handled by 3-5, not here.
//
//  All output fields (x y z intensity time ring) are preserved by name so
//  the downstream LIO Velodyne handler consumes it identically to the raw
//  topic. Run with enable_filtering:=false for the raw baseline (time-fix only).
//
//  Algorithmic detail for steps 0b, 3 and 6 lives in:
//     include/copernicus_localization/cloud_denoise.hpp
//     include/copernicus_localization/ground_segmentation.hpp
//     include/copernicus_localization/dynamic_removal.hpp
// =============================================================================

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "copernicus_localization/msg/environment_state.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "copernicus_localization/cloud_denoise.hpp"
#include "copernicus_localization/ground_segmentation.hpp"
#include "copernicus_localization/dynamic_removal.hpp"

// ---- Point types for I/O (matching Ouster, Hesai, Velodyne) -----------------
namespace cop
{
struct EIGEN_ALIGN16 OusterPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  std::uint32_t t;               // per-point timestamp in nanoseconds
  std::uint16_t reflectivity;
  std::uint16_t ring;
  std::uint16_t ambient;
  std::uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 HesaiPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;              // per-point Unix timestamp in seconds
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct EIGEN_ALIGN16 VelodynePoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;                    // per-point offset in seconds
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace cop

POINT_CLOUD_REGISTER_POINT_STRUCT(
  cop::OusterPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)(
    std::uint16_t, ring, ring)(std::uint16_t, ambient, ambient)(
    std::uint32_t, range, range))

POINT_CLOUD_REGISTER_POINT_STRUCT(
  cop::HesaiPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    double, timestamp, timestamp)(std::uint16_t, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(
  cop::VelodynePoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
    float, time, time)(std::uint16_t, ring, ring))

// ---- Internal working point type (float time in seconds + ring) -------------
// All processing stages operate on this type; conversion to/from sensor types
// happens at the I/O boundary of the callback.
namespace cop
{
struct EIGEN_ALIGN16 WorkPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;                    // seconds from scan start
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace cop

POINT_CLOUD_REGISTER_POINT_STRUCT(
  cop::WorkPoint,
  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(
    std::uint16_t, ring, ring))

using PointT = cop::WorkPoint;
using CloudT = pcl::PointCloud<PointT>;
using OusterCloud = pcl::PointCloud<cop::OusterPoint>;
using HesaiCloud = pcl::PointCloud<cop::HesaiPoint>;
using VelodyneCloud = pcl::PointCloud<cop::VelodynePoint>;

class CloudPreprocess : public rclcpp::Node
{
public:
  CloudPreprocess() : Node("cloud_preprocess")
  {
    input_topic_  = declare_parameter<std::string>("input_topic", "/velodyne_points");
    output_topic_ = declare_parameter<std::string>("output_topic", "/velodyne_points_processed");
    odom_topic_   = declare_parameter<std::string>("odom_topic", "/lio/odometry");

    enable_filtering_ = declare_parameter<bool>("enable_filtering", true);

    // 0b. Particulate denoise (DROR range-adaptive outlier removal). Off by
    //     default — only needed in snow/dust/rain; harmless but non-zero cost.
    enable_denoise_   = declare_parameter<bool>("enable_denoise", false);
    cop::DenoiseConfig nc;
    nc.azimuth_res_deg   = declare_parameter<double>("denoise_azimuth_res_deg", 0.20);
    nc.multiplier        = declare_parameter<double>("denoise_multiplier", 8.0);
    nc.min_search_radius = declare_parameter<double>("denoise_min_radius", 0.20);
    nc.max_search_radius = declare_parameter<double>("denoise_max_radius", 0.80);
    nc.min_neighbors     = declare_parameter<int>("denoise_min_neighbors", 2);
    nc.max_intensity     = declare_parameter<double>("denoise_max_intensity", -1.0);
    nc.min_range         = declare_parameter<double>("denoise_min_range", 0.0);
    denoiser_.configure(nc);

    // 1. Range crop (3D euclidean distance from the sensor).
    min_range_ = declare_parameter<double>("min_range", 0.8);
    max_range_ = declare_parameter<double>("max_range", 80.0);
    nominal_scan_period_ = declare_parameter<double>("scan_period", 0.10);
    // Move the output stamp to the first point's absolute time (required when the
    // driver stamps at scan END with negative per-point offsets, e.g. ROS1 velodyne).
    shift_stamp_ = declare_parameter<bool>("shift_stamp_to_first_point", true);

    // 2. Multi-return / intensity filter.
    enable_intensity_ = declare_parameter<bool>("enable_intensity_filter", false);
    min_intensity_    = declare_parameter<double>("min_intensity", 3.0);
    // Activates only if the cloud actually carries a dual-return/echo field.
    drop_secondary_returns_ = declare_parameter<bool>("drop_secondary_returns", true);

    // 3-4. Ground segmentation + lower-trunk band.
    enable_ground_seg_ = declare_parameter<bool>("enable_ground_seg", true);
    keep_ground_       = declare_parameter<bool>("keep_ground", true);
    band_min_          = declare_parameter<double>("band_min", 0.0);
    band_max_          = declare_parameter<double>("band_max", 3.0);
    cop::GroundConfig gc;
    gc.min_range      = declare_parameter<double>("ground_min_range", 1.0);
    gc.max_range      = declare_parameter<double>("ground_max_range", 60.0);
    gc.sensor_height  = declare_parameter<double>("sensor_height", 0.70);   // <= 0: auto
    gc.th_dist        = declare_parameter<double>("ground_th_dist", 0.15);
    gc.th_seeds       = declare_parameter<double>("ground_th_seeds", 0.40);
    gc.th_uprightness = declare_parameter<double>("ground_uprightness", 0.80);
    gc.th_flatness    = declare_parameter<double>("ground_flatness", 0.02);
    gc.elevation_tol  = declare_parameter<double>("ground_elevation_tol", 0.50);
    ground_.configure(gc);
    // Band fail-safe: below this fraction of populated bins accepted as ground the
    // ground model is untrustworthy, so the height band is skipped for the scan
    // (pass-through) instead of cutting at a fallback height.
    min_ground_bin_frac_ = declare_parameter<double>("min_ground_bin_frac", 0.15);

    // 5. Rigidity / foliage rejection (local structure tensor on non-ground).
    enable_rigidity_  = declare_parameter<bool>("enable_rigidity", false);
    rigidity_radius_  = declare_parameter<double>("rigidity_radius", 0.4);
    rigidity_min_nbrs_= declare_parameter<int>("rigidity_min_neighbors", 6);
    linearity_thresh_ = declare_parameter<double>("linearity_thresh", 0.55);
    planarity_thresh_ = declare_parameter<double>("planarity_thresh", 0.55);
    keep_sparse_      = declare_parameter<bool>("keep_sparse_points", true);

    // 6. Dynamic-object removal (Dynablox-style, keyframe rate, needs odometry).
    enable_dynamic_   = declare_parameter<bool>("enable_dynamic_removal", false);
    dynamic_debug_    = declare_parameter<bool>("dynamic_debug", false);
    lidar_in_body_z_  = declare_parameter<double>("lidar_in_body_z", 0.282);
    // Full LiDAR->IMU(body) extrinsic (LIO mapping.extrinsic_T / _R). The
    // launch fills these from the LIO config; empty -> (0,0,lidar_in_body_z), I.
    const auto ext_t = declare_parameter<std::vector<double>>("lidar_extrinsic_T", std::vector<double>{});
    const auto ext_r = declare_parameter<std::vector<double>>("lidar_extrinsic_R", std::vector<double>{});
    t_body_lidar_ = Eigen::Vector3f(0.f, 0.f, float(lidar_in_body_z_));
    if (ext_t.size() == 3) t_body_lidar_ = Eigen::Vector3f(ext_t[0], ext_t[1], ext_t[2]);
    R_body_lidar_.setIdentity();
    if (ext_r.size() == 9)
      R_body_lidar_ << ext_r[0], ext_r[1], ext_r[2], ext_r[3], ext_r[4], ext_r[5], ext_r[6], ext_r[7], ext_r[8];
    keyframe_dist_    = declare_parameter<double>("dynamic_keyframe_dist", 0.3);
    keyframe_time_    = declare_parameter<double>("dynamic_keyframe_time", 1.0);
    cop::DynamicConfig dc;
    dc.voxel_size         = declare_parameter<double>("dynamic_voxel_size", 0.40);
    dc.max_ray            = declare_parameter<double>("dynamic_max_ray", 40.0);
    dc.ever_free_count    = declare_parameter<int>("dynamic_ever_free_count", 3);
    dc.free_recency       = declare_parameter<int>("dynamic_free_recency", 30);
    dc.min_cluster_points = declare_parameter<int>("dynamic_min_cluster_points", 6);
    dc.max_cluster_extent = declare_parameter<double>("dynamic_max_cluster_extent", 4.0);
    dc.voxel_ttl_frames   = declare_parameter<int>("dynamic_voxel_ttl", 200);
    dynamic_.configure(dc);

    // Scene-adaptive filtering. The band / rigidity / DROR / dynamic steps are
    // forest defenses (canopy, foliage, dust, walkers) that delete exactly the sparse
    // structure a degenerate scene depends on (tunnel ceiling fixtures, distant posts,
    // bridge trusses). While the environment router reports SPARSE/POOR (or did so
    // within adaptive_hold_sec) only the range crop runs: every point reaches the LIO.
    adaptive_ = declare_parameter<bool>("adaptive_filtering", true);
    adaptive_hold_sec_ = declare_parameter<double>("adaptive_hold_sec", 5.0);
    adaptive_min_tier_ = declare_parameter<int>("adaptive_min_tier", 1);   // 1=SPARSE, 2=POOR
    env_topic_ = declare_parameter<std::string>("env_topic", "/environment/state");

    // Lidar is a real-time stream: never build a backlog of old scans. A deep
    // reliable queue can make the LIO consume stale clouds after one expensive
    // filtering spike, which looks like a severe Hz drop and breaks odometry.
    const auto cloud_qos = rclcpp::SensorDataQoS().keep_last(1);
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, cloud_qos);
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, cloud_qos,
      std::bind(&CloudPreprocess::callback, this, std::placeholders::_1));
    if (enable_dynamic_)
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).reliable(),
        std::bind(&CloudPreprocess::odomCallback, this, std::placeholders::_1));
    if (enable_dynamic_ && dynamic_debug_)
      dyn_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_ + "_dynamic", cloud_qos);
    if (enable_filtering_ && adaptive_)
      env_sub_ = create_subscription<copernicus_localization::msg::EnvironmentState>(
        env_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
        [this](const copernicus_localization::msg::EnvironmentState::ConstSharedPtr m) {
          if (m->environment_class >= adaptive_min_tier_) last_degraded_ = rclcpp::Time(m->header.stamp);
          have_env_ = true;
        });

    RCLCPP_INFO(get_logger(),
      "cloud_preprocess: %s -> %s | filtering=%s denoise=%s ground_seg=%s "
      "rigidity=%s intensity=%s dynamic=%s",
      input_topic_.c_str(), output_topic_.c_str(),
      enable_filtering_ ? "on" : "off",
      (enable_filtering_ && enable_denoise_) ? "on" : "off",
      (enable_filtering_ && enable_ground_seg_) ? "on" : "off",
      (enable_filtering_ && enable_rigidity_) ? "on" : "off",
      (enable_filtering_ && enable_intensity_) ? "on" : "off",
      (enable_filtering_ && enable_dynamic_) ? "on" : "off");
  }

private:
  // -------------------------------------------------------------------------
  void odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    world_frame_ = msg->header.frame_id;
    const auto & p = msg->pose.pose.position;
    const auto & q = msg->pose.pose.orientation;
    const rclcpp::Time stamp(msg->header.stamp);
    if (have_odom_)
    {
      const double dt = (stamp - odom_stamp_).seconds();
      if (dt > 1e-3 && dt < 0.5)
      {
        const Eigen::Vector3f t_new(p.x, p.y, p.z);
        const Eigen::Quaternionf q_new = Eigen::Quaternionf(q.w, q.x, q.y, q.z).normalized();
        odom_v_ = (t_new - odom_t_) / float(dt);
        const Eigen::AngleAxisf dq(odom_q_.conjugate() * q_new);   // body-frame increment
        odom_w_ = dq.axis() * dq.angle() / float(dt);
      }
    }
    odom_t_ = Eigen::Vector3f(p.x, p.y, p.z);
    odom_q_ = Eigen::Quaternionf(q.w, q.x, q.y, q.z).normalized();
    odom_stamp_ = stamp;
    have_odom_ = true;
  }

  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    const auto t0 = std::chrono::steady_clock::now();

    // Detect a dual-return / echo field and sensor format once.
    if (!fields_checked_) {
      has_return_field_ = hasReturnField(*msg);
      for (const auto & f : msg->fields) {
        if (f.name == "timestamp") is_hesai_ = true;
        if (f.name == "t") is_ouster_ = true;
      }
      fields_checked_ = true;
    }

    // ---- Read sensor cloud and convert to WorkPoint -------------------------
    CloudT::Ptr in(new CloudT);
    double hesai_base_time = 0.0;

    if (is_hesai_)
    {
      HesaiCloud::Ptr hs_in(new HesaiCloud);
      pcl::fromROSMsg(*msg, *hs_in);
      if (hs_in->empty()) return;
      in->points.reserve(hs_in->size());
      hesai_base_time = hs_in->points[0].timestamp;
      for (const auto & hp : hs_in->points)
      {
        PointT wp;
        wp.x = hp.x; wp.y = hp.y; wp.z = hp.z;
        wp.intensity = hp.intensity;
        wp.time = static_cast<float>(hp.timestamp - hesai_base_time);
        wp.ring = hp.ring;
        in->points.push_back(wp);
      }
    }
    else if (is_ouster_)
    {
      OusterCloud::Ptr os_in(new OusterCloud);
      pcl::fromROSMsg(*msg, *os_in);
      if (os_in->empty()) return;
      in->points.reserve(os_in->size());
      for (const auto & op : os_in->points)
      {
        PointT wp;
        wp.x = op.x; wp.y = op.y; wp.z = op.z;
        wp.intensity = op.intensity;
        wp.time = static_cast<float>(op.t) * 1e-9f;   // ns → seconds
        wp.ring = op.ring;
        in->points.push_back(wp);
      }
    }
    else
    {
      VelodyneCloud::Ptr vl_in(new VelodyneCloud);
      pcl::fromROSMsg(*msg, *vl_in);
      if (vl_in->empty()) return;
      in->points.reserve(vl_in->size());
      for (const auto & vp : vl_in->points)
      {
        PointT wp;
        wp.x = vp.x; wp.y = vp.y; wp.z = vp.z;
        wp.intensity = vp.intensity;
        wp.time = vp.time;
        wp.ring = vp.ring;
        in->points.push_back(wp);
      }
    }

    // ---- 0. Drop NaNs, find min per-point time, renormalise time -----------
    CloudT::Ptr valid(new CloudT);
    valid->points.reserve(in->size());
    float t_min = std::numeric_limits<float>::max();
    float t_max = -std::numeric_limits<float>::max();
    for (const auto & p : in->points)
    {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      if (std::isfinite(p.time))
      {
        if (p.time < t_min) t_min = p.time;
        if (p.time > t_max) t_max = p.time;
      }
      valid->points.push_back(p);
    }
    if (valid->empty()) return;
    const bool have_time_bounds =
      t_min != std::numeric_limits<float>::max() && t_max > t_min;
    if (!have_time_bounds) { t_min = 0.0f; t_max = static_cast<float>(nominal_scan_period_); }
    const float scan_duration = std::max(1e-3f, t_max - t_min);
    for (auto & p : valid->points)
      p.time = std::isfinite(p.time) ? (p.time - t_min) : 0.0f;

    CloudT::Ptr work = valid;

    // ---- 1..6 Spatial filtering (optional) ---------------------------------
    work_is_ground_.clear();   // (re)populated only when ground seg runs
    const rclcpp::Time scan_stamp(msg->header.stamp);
    const bool degraded = adaptive_ && have_env_ &&
      (scan_stamp - last_degraded_).seconds() < adaptive_hold_sec_;
    if (degraded != degraded_mode_)
    {
      RCLCPP_INFO(get_logger(), "cloud_preprocess: %s",
        degraded ? "degenerate scene -> range crop only (band/rigidity/denoise/dynamic bypassed)"
                 : "scene recovered -> full Stage-A filtering");
      degraded_mode_ = degraded;
    }
    if (enable_filtering_)
    {
      if (enable_denoise_ && !degraded)  work = denoiseFilter(work);         // 0b
      work = rangeFilter(work);                                              // 1
      if (enable_intensity_ && !degraded) work = intensityFilter(work);      // 2
      if (!degraded)
      {
        if (enable_ground_seg_) work = groundBandFilter(work);               // 3 + 4 (+5)
        else if (enable_rigidity_) work = rigidityFilter(work);              // 5 only
        if (enable_dynamic_)                                                 // 6
        {
          // the LIO's scan-start stamp (first point), which its odometry is stamped against
          const rclcpp::Time scan_begin = (shift_stamp_ && have_time_bounds && !is_hesai_)
            ? scan_stamp + rclcpp::Duration::from_seconds(double(t_min)) : scan_stamp;
          work = dynamicFilter(work, scan_begin, scan_duration);
        }
      }
      if (work->empty())
      {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "filtering removed every point; publishing time-fixed cloud for this scan");
        work = valid;
      }
      preserveScanEndTime(work, scan_duration);
    }

    // ---- 7. Convert WorkPoint → Output PointCloud2 and publish -------------
    sensor_msgs::msg::PointCloud2 out;
    if (is_hesai_)
    {
      HesaiCloud::Ptr hs_out(new HesaiCloud);
      hs_out->points.reserve(work->size());
      for (const auto & wp : work->points)
      {
        cop::HesaiPoint hp;
        hp.x = wp.x; hp.y = wp.y; hp.z = wp.z;
        hp.intensity = wp.intensity;
        hp.timestamp = hesai_base_time + double(t_min) + wp.time;   // back to absolute
        hp.ring = wp.ring;
        hs_out->points.push_back(hp);
      }
      pcl::toROSMsg(*hs_out, out);
    }
    else if (is_ouster_)
    {
      OusterCloud::Ptr os_out(new OusterCloud);
      os_out->points.reserve(work->size());
      for (const auto & wp : work->points)
      {
        cop::OusterPoint op;
        op.x = wp.x; op.y = wp.y; op.z = wp.z;
        op.intensity = wp.intensity;
        op.t = static_cast<std::uint32_t>(std::max(0.0f, wp.time) * 1e9f);   // seconds → ns
        op.reflectivity = 0;
        op.ring = wp.ring;
        op.ambient = 0;
        op.range = 0;
        os_out->points.push_back(op);
      }
      pcl::toROSMsg(*os_out, out);
    }
    else
    {
      VelodyneCloud::Ptr vl_out(new VelodyneCloud);
      vl_out->points.reserve(work->size());
      for (const auto & wp : work->points)
      {
        cop::VelodynePoint vp;
        vp.x = wp.x; vp.y = wp.y; vp.z = wp.z;
        vp.intensity = wp.intensity;
        vp.time = wp.time;
        vp.ring = wp.ring;
        vl_out->points.push_back(vp);
      }
      pcl::toROSMsg(*vl_out, out);
    }
    out.header = msg->header;
    // Per-point times were renormalised to start at 0, so the stamp must move to
    // the first point's absolute time. Velodyne drivers stamp at scan END with
    // negative offsets (t_min ~ -0.1 s): without this shift the LIO treats the scan
    // as [end, end+0.1] -> a full-scan LiDAR<->IMU offset in deskew/propagation.
    // Hesai carries ABSOLUTE per-point times and the LIO's hesai_handler measures
    // offsets from the FIRST point it receives while taking the header as scan start,
    // so the stamp must equal the first OUTPUT point's time — once the range crop
    // removes the operator/platform sector at the start of a sweep they differ by ms.
    if (shift_stamp_ && is_hesai_ && !work->empty())
      out.header.stamp = shiftStamp(msg->header.stamp,
        hesai_base_time + double(t_min) + double(work->points.front().time) - rclcpp::Time(msg->header.stamp).seconds());
    else if (shift_stamp_ && have_time_bounds)
      out.header.stamp = shiftStamp(msg->header.stamp, double(t_min));

    // Monotonicity guard: the LIO clears its buffers on any backward timestamp
    // jump ("lidar loop back"). Guarantee strictly non-decreasing output stamps.
    const rclcpp::Time cur_stamp(out.header.stamp);
    if (last_pub_stamp_.nanoseconds() > 0 && (last_pub_stamp_ - cur_stamp).seconds() > 1.0)
    {
      RCLCPP_WARN(get_logger(), "cloud_preprocess: time jumped back %.1f s (restart) -> guard reset",
        (last_pub_stamp_ - cur_stamp).seconds());
    }
    else if (last_pub_stamp_.nanoseconds() > 0 && cur_stamp < last_pub_stamp_)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "cloud_preprocess: suppressed non-monotonic stamp (%.9f < %.9f)",
        cur_stamp.seconds(), last_pub_stamp_.seconds());
      return;   // drop this scan to protect the LIO
    }
    last_pub_stamp_ = cur_stamp;

    out.is_dense = true;
    pub_->publish(out);

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
      "cloud_preprocess: %.1f ms/scan | pts %zu -> %zu | filtering=%s denoise=%s "
      "ground=%s rigidity=%s dynamic=%s",
      ms, in->size(), work->size(),
      enable_filtering_ ? "on" : "off",
      (enable_filtering_ && enable_denoise_) ? "on" : "off",
      (enable_filtering_ && enable_ground_seg_) ? "on" : "off",
      (enable_filtering_ && enable_rigidity_) ? "on" : "off",
      (enable_filtering_ && enable_dynamic_) ? "on" : "off");
  }

  // 0b. Particulate denoise (DROR). Order-preserving: builds a keep mask over
  //     the full cloud and emits survivors in original scan order (the LIO
  //     scan-end-time invariant — never reorder points). Runs before everything
  //     so snow/dust/rain specks never reach the ground fit or front-end.
  CloudT::Ptr denoiseFilter(const CloudT::Ptr & cloud)
  {
    std::vector<uint8_t> keep;
    const size_t nrem = denoiser_.denoise(cloud->points, keep);
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size() - nrem);
    for (size_t i = 0; i < cloud->size(); ++i)
      if (keep[i]) out->points.push_back(cloud->points[i]);
    if (dynamic_debug_)
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "denoise: removed %zu / %zu pts (%.1f%%)",
        nrem, cloud->size(), cloud->empty() ? 0.0 : 100.0 * nrem / cloud->size());
    return out;
  }

  // 1. Euclidean range gate: removes invalid-near (ego/mount) and far-sparse.
  CloudT::Ptr rangeFilter(const CloudT::Ptr & cloud) const
  {
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    const double r2min = min_range_ * min_range_;
    const double r2max = max_range_ * max_range_;
    for (const auto & p : cloud->points)
    {
      const double r2 = double(p.x) * p.x + double(p.y) * p.y + double(p.z) * p.z;
      if (r2 >= r2min && r2 <= r2max) out->points.push_back(p);
    }
    return out;
  }

  // 2. Multi-return / intensity filter. Drops weak returns (foliage/dust/edge
  //    points) below min_intensity. If the driver runs dual-return AND exposes
  //    an echo field, the weak-echo points are the low-intensity duplicates at
  //    the same bearing, so the same intensity gate removes them at zero extra
  //    cost; with a true return-type field present we additionally log it once.
  CloudT::Ptr intensityFilter(const CloudT::Ptr & cloud) const
  {
    if (has_return_field_ && drop_secondary_returns_)
      RCLCPP_INFO_ONCE(get_logger(),
        "intensity filter: dual-return/echo field present; weak echoes dropped "
        "via the intensity gate (min_intensity=%.1f)", min_intensity_);
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    const float thr = static_cast<float>(min_intensity_);
    for (const auto & p : cloud->points)
      if (p.intensity >= thr) out->points.push_back(p);
    return out;
  }

  void preserveScanEndTime(const CloudT::Ptr & cloud, float scan_duration) const
  {
    if (cloud->empty() || !std::isfinite(scan_duration) || scan_duration <= 0.0f) return;
    const float last_time = cloud->points.back().time;
    if (std::isfinite(last_time) && last_time >= 0.95f * scan_duration) return;

    PointT anchor = cloud->points.back();
    anchor.time = scan_duration;
    cloud->points.push_back(anchor);
  }

  // 3 + 4 (+5). Patchwork++-style ground segmentation, then the lower-trunk
  //    band passthrough built on the *fitted* ground, then (optionally) the
  //    rigidity test applied to the non-ground band only.
  //
  //    CRITICAL: points are emitted in their ORIGINAL scan order. the LIO
  //    derives the scan-end time from the LAST point's per-point timestamp
  //    (laserMapping.cpp: points.back().curvature); reordering the cloud would
  //    feed it an arbitrary end time -> IMU/LiDAR desync -> tracking fails.
  CloudT::Ptr groundBandFilter(const CloudT::Ptr & cloud)
  {
    const size_t n = cloud->size();
    std::vector<uint8_t> is_ground;
    ground_.segment(cloud->points, is_ground);

    if (ground_.heightReady() && !height_logged_)
    {
      RCLCPP_INFO(get_logger(), "ground: sensor height %.2f m (%s)", ground_.sensorHeight(),
                  ground_.config().sensor_height > 0.0 ? "configured" : "auto-estimated");
      height_logged_ = true;
    }
    // Fail-safe: no trustworthy ground model (auto height still calibrating, or too
    // few bins accepted) -> never cut by height; keep everything, ground mask intact.
    const double gfrac = ground_.groundBinFraction();
    if (!ground_.heightReady() || gfrac < min_ground_bin_frac_)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "ground model unreliable (%.0f%% bins accepted, height %s) -> band filter skipped",
        100.0 * gfrac, ground_.heightReady() ? "ready" : "calibrating");
      work_is_ground_ = is_ground;
      return cloud;
    }

    // Per-point keep decision; collect the non-ground band subset for rigidity.
    std::vector<uint8_t> keep(n, 0u);
    CloudT::Ptr band(new CloudT);
    std::vector<int> band_idx;
    band->points.reserve(n);
    band_idx.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
      if (is_ground[i]) { if (keep_ground_) keep[i] = 1u; continue; }
      const auto & p = cloud->points[i];
      const float h = p.z - ground_.groundHeightAtPoint(i, p.x, p.y);
      // band_max <= 0 disables the upper cap (keep ceilings / superstructure)
      if (h >= band_min_ && (band_max_ <= 0.0 || h <= band_max_)) { band->points.push_back(p); band_idx.push_back(int(i)); }
    }

    // 5. Rigidity on the non-ground band (trunks survive, foliage drops).
    if (enable_rigidity_)
    {
      std::vector<uint8_t> band_keep;
      rigidityMask(band, band_keep);
      for (size_t b = 0; b < band_idx.size(); ++b)
        if (band_keep[b]) keep[band_idx[b]] = 1u;
    }
    else
    {
      for (int idx : band_idx) keep[idx] = 1u;
    }

    // Emit in original order; build the ground mask in parity for dynamic removal.
    CloudT::Ptr out(new CloudT);
    out->points.reserve(n);
    work_is_ground_.clear();
    work_is_ground_.reserve(n);
    for (size_t i = 0; i < n; ++i)
      if (keep[i]) { out->points.push_back(cloud->points[i]); work_is_ground_.push_back(is_ground[i]); }
    return out;
  }

  // 5. Rigidity classification via local structure tensor. Keep points whose
  //    neighbourhood is linear (trunks) or planar; drop scattered/volumetric
  //    neighbourhoods (wind-driven foliage). Fills `keep` (1 = keep), aligned to
  //    `cloud`.
  //
  //    VOXEL-BASED (fast): accumulate first/second moments per voxel of edge
  //    `rigidity_radius`, then classify each occupied voxel from the combined
  //    moments of its 3x3x3 neighbourhood (one eigen-decomposition per voxel,
  //    cached), and label all points in that voxel. This is O(N) and ~50-100x
  //    faster than the previous per-point kd-tree radius search (~1 s/scan),
  //    which made full-rate localization impossible.
  void rigidityMask(const CloudT::Ptr & cloud, std::vector<uint8_t> & keep) const
  {
    const size_t n = cloud->size();
    keep.assign(n, 0u);
    if (n < static_cast<size_t>(rigidity_min_nbrs_)) { std::fill(keep.begin(), keep.end(), 1u); return; }

    const double inv = 1.0 / rigidity_radius_;
    auto packk = [](int64_t ix, int64_t iy, int64_t iz) -> int64_t {
      return ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF);
    };
    struct Mom { int n = 0; double sx=0,sy=0,sz=0, xx=0,xy=0,xz=0,yy=0,yz=0,zz=0; };

    std::unordered_map<int64_t, Mom> vox;
    vox.reserve(n);
    std::vector<int64_t> cx(n), cy(n), cz(n);
    for (size_t i = 0; i < n; ++i)
    {
      const auto & p = cloud->points[i];
      const int64_t ix = int64_t(std::floor(p.x * inv));
      const int64_t iy = int64_t(std::floor(p.y * inv));
      const int64_t iz = int64_t(std::floor(p.z * inv));
      cx[i] = ix; cy[i] = iy; cz[i] = iz;
      Mom & m = vox[packk(ix, iy, iz)];
      m.n++;
      m.sx += p.x; m.sy += p.y; m.sz += p.z;
      m.xx += double(p.x)*p.x; m.xy += double(p.x)*p.y; m.xz += double(p.x)*p.z;
      m.yy += double(p.y)*p.y; m.yz += double(p.y)*p.z; m.zz += double(p.z)*p.z;
    }

    std::unordered_map<int64_t, uint8_t> cls;       // voxel -> keep(1)/drop(0)
    cls.reserve(vox.size());
    for (size_t i = 0; i < n; ++i)
    {
      const int64_t k = packk(cx[i], cy[i], cz[i]);
      auto c = cls.find(k);
      if (c != cls.end()) { keep[i] = c->second; continue; }

      Mom a;                                         // 3x3x3 neighbourhood
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz)
          {
            auto it = vox.find(packk(cx[i]+dx, cy[i]+dy, cz[i]+dz));
            if (it == vox.end()) continue;
            const Mom & m = it->second;
            a.n += m.n; a.sx += m.sx; a.sy += m.sy; a.sz += m.sz;
            a.xx += m.xx; a.xy += m.xy; a.xz += m.xz;
            a.yy += m.yy; a.yz += m.yz; a.zz += m.zz;
          }

      uint8_t kp;
      if (a.n < rigidity_min_nbrs_) kp = keep_sparse_ ? 1u : 0u;
      else
      {
        const double N = a.n, mx = a.sx/N, my = a.sy/N, mz = a.sz/N;
        Eigen::Matrix3d cov;
        cov(0,0) = a.xx/N - mx*mx; cov(0,1) = a.xy/N - mx*my; cov(0,2) = a.xz/N - mx*mz;
        cov(1,1) = a.yy/N - my*my; cov(1,2) = a.yz/N - my*mz; cov(2,2) = a.zz/N - mz*mz;
        cov(1,0) = cov(0,1); cov(2,0) = cov(0,2); cov(2,1) = cov(1,2);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Vector3d ev = es.eigenvalues();        // ascending
        const double e2 = std::max(ev[2], 1e-9);
        const double e1 = std::max(ev[1], 0.0);
        const double e0 = std::max(ev[0], 0.0);
        const double linearity = (e2 - e1) / e2;
        const double planarity = (e1 - e0) / e2;
        kp = (linearity >= linearity_thresh_ || planarity >= planarity_thresh_) ? 1u : 0u;
      }
      cls[k] = kp;
      keep[i] = kp;
    }
  }

  // Order-preserving rigidity filter (used when ground seg is off).
  CloudT::Ptr rigidityFilter(const CloudT::Ptr & cloud) const
  {
    std::vector<uint8_t> keep;
    rigidityMask(cloud, keep);
    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i)
      if (keep[i]) out->points.push_back(cloud->points[i]);
    return out;
  }

  // 6. Dynablox-style dynamic-object removal. Transforms the cloud to the world
  //    frame via the latest odometry and drops confirmed dynamic points. The
  //    cloud is *classified on every scan* (cheap voxel lookup) so movers never
  //    leak through; the temporal voxel map is *updated only at keyframe rate*
  //    (the expensive ray casting). No-op until odometry is available.
  CloudT::Ptr dynamicFilter(const CloudT::Ptr & cloud, const rclcpp::Time & scan_stamp, float scan_duration)
  {
    if (!have_odom_) return cloud;

    // Keyframe gate decides whether to UPDATE the map this scan (not whether to
    // filter). Classification still runs on every scan.
    const rclcpp::Time now = this->get_clock()->now();
    const bool first = !dyn_initialised_;
    const double moved = first ? 1e9 : (odom_t_ - last_dyn_t_).norm();
    const double dt    = first ? 1e9 : (now - last_dyn_time_).seconds();
    const bool keyframe = first || moved >= keyframe_dist_ || dt >= keyframe_time_;
    if (keyframe) { last_dyn_t_ = odom_t_; last_dyn_time_ = now; dyn_initialised_ = true; }

    // The latest /Odometry is the PREVIOUS scan's end pose; extrapolate it to this
    // scan's mid-time with the last odom velocity (constant-twist model) so a moving
    // platform does not smear static structure into "free" voxels.
    const float dt_x = std::clamp(float((scan_stamp - odom_stamp_).seconds()) + 0.5f * scan_duration, 0.f, 0.3f);
    const Eigen::Vector3f t_wb = odom_t_ + odom_v_ * dt_x;
    const Eigen::Vector3f wdt = odom_w_ * dt_x;
    const Eigen::Quaternionf q_wb = (wdt.norm() > 1e-6f)
      ? (odom_q_ * Eigen::Quaternionf(Eigen::AngleAxisf(wdt.norm(), wdt.normalized()))).normalized()
      : odom_q_;

    // world_T_lidar = world_T_body * body_T_lidar (full extrinsic R, T).
    const Eigen::Matrix3f R = q_wb.toRotationMatrix();
    const Eigen::Matrix3f R_wl = R * R_body_lidar_;
    const Eigen::Vector3f sensor_world = R * t_body_lidar_ + t_wb;

    std::vector<Eigen::Vector3f> world(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i)
    {
      const auto & p = cloud->points[i];
      world[i] = R_wl * Eigen::Vector3f(p.x, p.y, p.z) + sensor_world;
    }

    const std::vector<uint8_t> * ground =
      (work_is_ground_.size() == cloud->size()) ? &work_is_ground_ : nullptr;

    std::vector<uint8_t> is_dyn;
    dynamic_.process(world, sensor_world, is_dyn, keyframe, ground);

    CloudT::Ptr out(new CloudT);
    out->points.reserve(cloud->size());
    size_t nrem = 0;
    for (size_t i = 0; i < cloud->size(); ++i)
    {
      if (is_dyn[i]) ++nrem;
      else out->points.push_back(cloud->points[i]);
    }

    if (dynamic_debug_)
    {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "dynamic: removed %zu / %zu pts (keyframe=%d, voxels=%zu)",
        nrem, cloud->size(), keyframe ? 1 : 0, dynamic_.mapSize());
      dyn_pub_->publish(toWorldMsg(world, is_dyn));   // removed points, world frame
    }
    return out;
  }

  // PointCloud2 of the removed (dynamic) points in the world frame, so it can be
  // overlaid on /lio/cloud_registered in RViz (debug only).
  sensor_msgs::msg::PointCloud2 toWorldMsg(
      const std::vector<Eigen::Vector3f> & world, const std::vector<uint8_t> & is_dyn)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr w(new pcl::PointCloud<pcl::PointXYZ>);
    w->points.reserve(world.size());
    for (size_t i = 0; i < is_dyn.size(); ++i)
      if (is_dyn[i]) { pcl::PointXYZ q; q.x = world[i].x(); q.y = world[i].y(); q.z = world[i].z(); w->points.push_back(q); }
    sensor_msgs::msg::PointCloud2 m;
    pcl::toROSMsg(*w, m);
    m.header.frame_id = world_frame_;
    m.header.stamp = this->get_clock()->now();
    return m;
  }

  static bool hasReturnField(const sensor_msgs::msg::PointCloud2 & msg)
  {
    for (const auto & f : msg.fields)
      if (f.name == "return_type" || f.name == "echo" || f.name == "reflectivity_return")
        return true;
    return false;
  }

  static builtin_interfaces::msg::Time shiftStamp(
    const builtin_interfaces::msg::Time & in, double shift_s)
  {
    int64_t ns = static_cast<int64_t>(in.sec) * 1000000000LL + in.nanosec;
    ns += static_cast<int64_t>(std::llround(shift_s * 1e9));
    builtin_interfaces::msg::Time out;
    out.sec = static_cast<int32_t>(ns / 1000000000LL);
    out.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
    return out;
  }

  // Parameters / topics
  std::string input_topic_, output_topic_, odom_topic_;
  std::string world_frame_ = "odom";   // frame of the LIO odometry
  bool enable_filtering_;
  bool enable_denoise_;
  double min_range_, max_range_, nominal_scan_period_;
  bool shift_stamp_;
  bool enable_intensity_, drop_secondary_returns_;
  double min_intensity_;
  bool enable_ground_seg_, keep_ground_;
  double band_min_, band_max_;
  bool enable_rigidity_;
  double rigidity_radius_;
  int rigidity_min_nbrs_;
  double linearity_thresh_, planarity_thresh_;
  bool keep_sparse_;
  bool enable_dynamic_, dynamic_debug_;
  double lidar_in_body_z_, keyframe_dist_, keyframe_time_;
  Eigen::Vector3f t_body_lidar_{0, 0, 0};
  Eigen::Matrix3f R_body_lidar_ = Eigen::Matrix3f::Identity();
  double min_ground_bin_frac_;
  bool adaptive_;
  double adaptive_hold_sec_;
  int adaptive_min_tier_;
  std::string env_topic_;

  // Stage modules
  cop::ParticulateDenoiser denoiser_;
  cop::GroundSegmenter ground_;
  cop::DynamicRemover dynamic_;

  // Runtime state
  std::vector<uint8_t> work_is_ground_;   // ground mask aligned to ground-seg output
  bool fields_checked_ = false, has_return_field_ = false;
  bool is_hesai_ = false, is_ouster_ = false;
  bool have_odom_ = false, dyn_initialised_ = false;
  Eigen::Vector3f odom_t_{0, 0, 0}, last_dyn_t_{0, 0, 0};
  Eigen::Quaternionf odom_q_{1, 0, 0, 0};
  Eigen::Vector3f odom_v_{0, 0, 0}, odom_w_{0, 0, 0};   // world lin. vel, body ang. vel
  rclcpp::Time odom_stamp_{0, 0, RCL_ROS_TIME};
  bool height_logged_ = false;
  bool have_env_ = false, degraded_mode_ = false;
  rclcpp::Time last_degraded_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_dyn_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_pub_stamp_{0, 0, RCL_ROS_TIME};   // for monotonicity guard

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dyn_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<copernicus_localization::msg::EnvironmentState>::SharedPtr env_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudPreprocess>());
  rclcpp::shutdown();
  return 0;
}
