// =============================================================================
//  Stage D: loop-closure DETECTOR        (GNSS-denied multi-terrain loc.)
// -----------------------------------------------------------------------------
//  Subscribes : /lio/odometry      (LIO IMU pose; keyframe gating + frames)
//               /lio/cloud_registered (LIO scan, world frame)
//               /environment/state (optional: skip detection in POOR scenes)
//  Publishes  : /loop_constraints  (LoopConstraint, consumed by the Stage-E backend)
//               /loop_markers      (visualization_msgs/MarkerArray: loop edges)
//
//  Built so it cannot inject garbage into the graph — degenerate scenes (runway,
//  tunnel, bridge deck) make Scan Context AND ICP agree on WRONG places, so every
//  candidate passes a chain of independent vetoes:
//
//    0. SYNC        — each cloud is paired with the /Odometry pose of the SAME
//       stamp (both are the LIO's lidar_end_time), never "the latest odom".
//    1. CANDIDATE   — Scan Context (height-based) top-K by ring-key + column-shift
//       distance. Candidates must be at least `min_loop_path` metres of TRAVEL
//       back (a keyframe count is not a distance: 30 kf on a straight runway is
//       30 m of identical asphalt), and odometrically reachable within the drift
//       budget. Low-structure descriptors (flat ground only) never search.
//    2. VERIFY      — GICP of the query keyframe against a submap around the match,
//       from two inits (SC yaw, odometry). Accept on inlier OVERLAP + inlier RMSE.
//    3. DRIFT GATE  — the registration must agree with odometry within
//       drift_ratio * path_length + floor: kills "sliding" aliases along a corridor
//       whose ICP converges anywhere along the unobservable axis.
//    4. OBSERVABILITY — point-to-plane information of the final alignment, eigen-
//       analysed (same normalisation as /lio/localizability). Too many
//       degenerate axes -> reject; otherwise the information sent to the backend is
//       ~0 along degenerate directions, so a tunnel loop only constrains the
//       cross-section, never the tunnel axis.
//    5. PCM         — pairwise consistency of loop TRANSFORMS through odometry
//       (Z_a * O(qa->qb) == O(ma->mb) * Z_b), greedy max clique. Loops that drop
//       out of the clique are RETRACTED from the backend.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "copernicus_localization/msg/loop_constraint.hpp"
#include "copernicus_localization/msg/environment_state.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>

#include "copernicus_localization/scan_context.hpp"

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;
using LoopMsg = copernicus_localization::msg::LoopConstraint;
using EnvMsg = copernicus_localization::msg::EnvironmentState;
using Mat6 = Eigen::Matrix<double, 6, 6>;

class LoopClosureNode : public rclcpp::Node
{
public:
  LoopClosureNode() : Node("loop_closure_node")
  {
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/lio/cloud_registered");
    odom_topic_  = declare_parameter<std::string>("odom_topic", "/lio/odometry");
    env_topic_   = declare_parameter<std::string>("env_topic", "/environment/state");

    kf_dist_ = declare_parameter<double>("kf_dist", 1.0);
    kf_time_ = declare_parameter<double>("kf_time", 2.0);
    kf_min_move_ = declare_parameter<double>("kf_min_move", 0.2);   // time keyframes need this much motion
    work_radius_ = declare_parameter<double>("work_radius", 60.0);

    cop::ScanContextConfig sc;
    sc.num_rings   = declare_parameter<int>("sc_rings", 20);
    sc.num_sectors = declare_parameter<int>("sc_sectors", 60);
    sc.max_radius  = declare_parameter<double>("sc_max_radius", 60.0);
    sc.height_offset = declare_parameter<double>("sc_height_offset", 2.0);
    scd_.configure(sc);
    struct_height_   = declare_parameter<double>("structure_height", 0.5);
    min_struct_bins_ = declare_parameter<int>("min_structured_bins", 15);
    min_struct_frac_ = declare_parameter<double>("min_structure_frac", 0.05);

    top_k_          = declare_parameter<int>("top_k", 10);
    sc_thresh_      = declare_parameter<double>("sc_dist_thresh", 0.40);
    ringkey_gate_   = declare_parameter<double>("ringkey_gate", 1.0);
    min_loop_path_  = declare_parameter<double>("min_loop_path", 50.0);

    icp_leaf_      = declare_parameter<double>("icp_leaf", 0.5);
    icp_max_pts_   = declare_parameter<int>("icp_max_points", 4000);
    icp_max_corr_  = declare_parameter<double>("icp_max_corr", 3.0);
    icp_iters_     = declare_parameter<int>("icp_iters", 30);
    submap_kf_     = declare_parameter<int>("submap_kf", 3);
    max_loop_trans_= declare_parameter<double>("max_loop_translation", 5.0);
    inlier_dist_   = declare_parameter<double>("inlier_dist", 0.5);
    min_overlap_   = declare_parameter<double>("min_overlap", 0.5);
    max_rmse_      = declare_parameter<double>("max_rmse", 0.25);

    drift_ratio_     = declare_parameter<double>("drift_ratio", 0.05);
    drift_floor_     = declare_parameter<double>("drift_floor", 2.0);
    drift_rot_ratio_ = declare_parameter<double>("drift_rot_ratio", 0.005);   // rad per m
    drift_rot_floor_ = declare_parameter<double>("drift_rot_floor", 0.25);

    degen_trans_thresh_ = declare_parameter<double>("degen_trans_thresh", 0.02);
    degen_rot_thresh_   = declare_parameter<double>("degen_rot_thresh", 0.02);
    max_degen_trans_    = declare_parameter<int>("max_degenerate_trans_axes", 1);
    max_degen_rot_      = declare_parameter<int>("max_degenerate_rot_axes", 1);
    loop_sigma_xyz_     = declare_parameter<double>("loop_sigma_xyz", 0.10);
    loop_sigma_rot_     = declare_parameter<double>("loop_sigma_rot", 0.02);

    pcm_trans_ratio_ = declare_parameter<double>("pcm_trans_ratio", 0.03);
    pcm_trans_floor_ = declare_parameter<double>("pcm_trans_floor", 1.0);
    pcm_rot_ratio_   = declare_parameter<double>("pcm_rot_ratio", 0.003);
    pcm_rot_floor_   = declare_parameter<double>("pcm_rot_floor", 0.10);
    pcm_min_clique_  = declare_parameter<int>("pcm_min_clique", 2);
    skip_in_poor_    = declare_parameter<bool>("skip_in_poor", false);
    loop_debug_      = declare_parameter<bool>("loop_debug", false);

    // CONSUMER subscriptions are best-effort + shallow so this keyframe-rate node
    // can NEVER back-pressure the LIO front-end. Loop constraints are reliable
    // with a deep queue: each one is graph information and must not be dropped.
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).best_effort(),
      std::bind(&LoopClosureNode::cloudCb, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
      std::bind(&LoopClosureNode::odomCb, this, std::placeholders::_1));
    if (skip_in_poor_)
      env_sub_ = create_subscription<EnvMsg>(
        env_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
        [this](const EnvMsg::ConstSharedPtr m) { env_class_ = m->environment_class; });
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/loop_markers", rclcpp::QoS(rclcpp::KeepLast(5)).reliable());
    loop_pub_ = create_publisher<LoopMsg>(
      "/loop_constraints", rclcpp::QoS(rclcpp::KeepLast(100)).reliable());

    accum_.reset(new CloudT);
    RCLCPP_INFO(get_logger(),
      "loop_closure_node: SC %dx%d r=%.0f | min_path=%.0fm drift=%.2f*L+%.1f | overlap>=%.2f "
      "rmse<=%.2f | degen<=%d axes | pcm clique>=%d | debug=%s",
      sc.num_rings, sc.num_sectors, sc.max_radius, min_loop_path_, drift_ratio_, drift_floor_,
      min_overlap_, max_rmse_, max_degen_trans_, pcm_min_clique_, loop_debug_ ? "on" : "off");
  }

private:
  struct OdomSample { int64_t ns; Eigen::Isometry3d pose; double path; };
  struct KF { Eigen::Isometry3d pose; double path; cop::ScanContextDesc desc; CloudT::Ptr cloud;
              builtin_interfaces::msg::Time stamp; bool structured; };
  struct Loop
  {
    int q, m; uint32_t id;
    Eigen::Isometry3d Z;                 // T_match_query (registration)
    Mat6 info;                           // GTSAM tangent order (rot, trans), query frame
    Eigen::Matrix3d Pobs;                // projector onto observable translation dirs (query frame)
    float sc_dist; double overlap, rmse; int degen_axes;
    bool published = false;
  };

  static int64_t toNs(const builtin_interfaces::msg::Time & t)
  { return int64_t(t.sec) * 1000000000LL + int64_t(t.nanosec); }

  static geometry_msgs::msg::Pose toPoseMsg(const Eigen::Isometry3d & T)
  {
    geometry_msgs::msg::Pose p;
    const Eigen::Quaterniond q(T.rotation());
    p.position.x = T.translation().x(); p.position.y = T.translation().y(); p.position.z = T.translation().z();
    p.orientation.w = q.w(); p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z();
    return p;
  }

  // ---- stamp synchronisation ------------------------------------------------
  void odomCb(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const auto & p = msg->pose.pose.position;
    const auto & o = msg->pose.pose.orientation;
    OdomSample s;
    s.ns = toNs(msg->header.stamp);
    s.pose = Eigen::Isometry3d::Identity();
    s.pose.linear() = Eigen::Quaterniond(o.w, o.x, o.y, o.z).normalized().toRotationMatrix();
    s.pose.translation() = Eigen::Vector3d(p.x, p.y, p.z);
    if (!odom_buf_.empty() && odom_buf_.back().ns - s.ns > 1000000000LL)  // >1 s back: restart
    {
      RCLCPP_WARN(get_logger(), "loop_closure: time jumped back (restart) -> odom/cloud buffers reset");
      odom_buf_.clear();
      pending_.clear();
    }
    if (!odom_buf_.empty() && s.ns <= odom_buf_.back().ns) return;      // non-monotonic
    path_total_ += odom_buf_.empty() ? 0.0
      : (s.pose.translation() - odom_buf_.back().pose.translation()).norm();
    s.path = path_total_;
    odom_buf_.push_back(s);
    while (odom_buf_.size() > 400) odom_buf_.pop_front();
    drainPending();
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    world_frame_ = msg->header.frame_id;
    pending_.push_back(msg);
    while (pending_.size() > 10) pending_.pop_front();
    drainPending();
  }

  // Pair each pending cloud with the odom sample of the same stamp. Wait while the
  // odom may still arrive; fall back to the nearest sample (<=60 ms) once odom is
  // clearly past the cloud; otherwise drop the cloud.
  void drainPending()
  {
    while (!pending_.empty() && !odom_buf_.empty())
    {
      const auto & msg = pending_.front();
      const int64_t t = toNs(msg->header.stamp);
      const OdomSample * best = nullptr; int64_t bd = INT64_MAX;
      for (auto it = odom_buf_.rbegin(); it != odom_buf_.rend(); ++it)
      {
        const int64_t d = std::llabs(it->ns - t);
        if (d < bd) { bd = d; best = &(*it); }
        if (it->ns < t - 200000000LL) break;
      }
      const bool exact = best && bd <= 5000000LL;                       // 5 ms
      const bool odom_past = odom_buf_.back().ns > t + 300000000LL;     // 0.3 s later
      if (!exact && !odom_past) return;                                 // wait for odom
      if (exact || (best && bd <= 60000000LL)) processCloud(*msg, *best);
      pending_.pop_front();
    }
  }

  void processCloud(const sensor_msgs::msg::PointCloud2 & msg, const OdomSample & od)
  {
    CloudT scan; pcl::fromROSMsg(msg, scan);
    const Eigen::Vector3d t = od.pose.translation();
    const double r2 = work_radius_ * work_radius_;
    for (const auto & p : scan.points)
    {
      const double dx = p.x - t.x(), dy = p.y - t.y();
      if (dx * dx + dy * dy <= r2) accum_->points.push_back(p);
    }

    const rclcpp::Time now = msg.header.stamp;
    const bool first = !kf_init_;
    const double moved = first ? 1e9 : (t - last_kf_t_).norm();
    const double dt    = first ? 1e9 : (now - last_kf_time_).seconds();
    if (!(first || moved >= kf_dist_ || dt >= kf_time_)) return;
    if (!first && moved < kf_min_move_)
    {
      // parked: no new keyframe (the database would otherwise grow every kf_time
      // forever); drop the accumulation so it stays bounded too
      last_kf_time_ = now;
      accum_->points.clear();
      return;
    }
    last_kf_t_ = t; last_kf_time_ = now; kf_init_ = true;

    if (accum_->size() >= 100) addKeyframe(msg.header.stamp, od);
    accum_->points.clear();
    publishMarkers(msg.header.stamp);
  }

  void addKeyframe(const builtin_interfaces::msg::Time & stamp, const OdomSample & od)
  {
    // accumulated world cloud -> this keyframe's body frame
    const Eigen::Isometry3f Tbw = od.pose.inverse().cast<float>();
    CloudT::Ptr body(new CloudT); body->points.reserve(accum_->size());
    for (const auto & p : accum_->points)
    {
      const Eigen::Vector3f b = Tbw * Eigen::Vector3f(p.x, p.y, p.z);
      PointT q; q.x = b.x(); q.y = b.y(); q.z = b.z(); q.intensity = p.intensity;
      body->points.push_back(q);
    }

    KF kf;
    kf.pose = od.pose;
    kf.path = od.path;
    kf.stamp = stamp;
    kf.desc = scd_.make(body->points);
    int nstruct = 0; double frac = 0.0;
    scd_.structure(kf.desc, float(struct_height_), nstruct, frac);
    kf.structured = nstruct >= min_struct_bins_ && frac >= min_struct_frac_;
    kf.cloud = downsample(body, icp_max_pts_);

    const int idx = int(db_.size());
    db_.push_back(kf);
    detect(idx);
  }

  CloudT::Ptr downsample(const CloudT::Ptr & in, int cap) const
  {
    CloudT::Ptr ds(new CloudT);
    pcl::VoxelGrid<PointT> vg;
    vg.setLeafSize(icp_leaf_, icp_leaf_, icp_leaf_);
    vg.setInputCloud(in);
    vg.filter(*ds);
    // stride-subsample to the cap (preserves coverage); keep PCL metadata in sync
    if (int(ds->size()) > cap)
    {
      CloudT::Ptr capped(new CloudT);
      capped->points.reserve(cap);
      const double stride = double(ds->size()) / double(cap);
      for (int i = 0; i < cap; ++i) capped->points.push_back(ds->points[size_t(i * stride)]);
      capped->width = capped->points.size(); capped->height = 1; capped->is_dense = ds->is_dense;
      ds.swap(capped);
    }
    return ds;
  }

  // submap of keyframes [m-k, m+k] expressed in keyframe m's frame
  CloudT::Ptr submap(int m) const
  {
    CloudT::Ptr out(new CloudT);
    const Eigen::Isometry3d Tm_inv = db_[m].pose.inverse();
    const int lo = std::max(0, m - submap_kf_), hi = std::min(int(db_.size()) - 1, m + submap_kf_);
    for (int j = lo; j <= hi; ++j)
    {
      const Eigen::Isometry3f T = (Tm_inv * db_[j].pose).cast<float>();
      for (const auto & p : db_[j].cloud->points)
      {
        const Eigen::Vector3f v = T * Eigen::Vector3f(p.x, p.y, p.z);
        PointT q; q.x = v.x(); q.y = v.y(); q.z = v.z(); q.intensity = p.intensity;
        out->points.push_back(q);
      }
    }
    out->width = out->points.size(); out->height = 1; out->is_dense = true;
    return downsample(out, 3 * icp_max_pts_);
  }

  void reject(int qi, int mi, const char * why, double a = 0.0, double b = 0.0)
  {
    ++n_rejected_;
    if (loop_debug_)
      RCLCPP_INFO(get_logger(), "reject kf%d<->kf%d: %s (%.3f / %.3f)", qi, mi, why, a, b);
  }

  // ---- the gates ------------------------------------------------------------
  void detect(int qi)
  {
    const KF & Q = db_[qi];
    if (!Q.structured) return;                                         // flat / featureless
    if (skip_in_poor_ && env_class_ == EnvMsg::POOR) return;

    // 1. CANDIDATE: travel + drift-reachability + ring-key, then full SC distance
    std::vector<std::pair<float, int>> cand;
    for (int m = 0; m < qi; ++m)
    {
      const KF & M = db_[m];
      const double L = Q.path - M.path;
      if (L < min_loop_path_) break;                                   // path is monotonic
      if (!M.structured) continue;
      const double gap = (Q.pose.translation() - M.pose.translation()).norm();
      if (gap > drift_ratio_ * L + drift_floor_ + max_loop_trans_) continue;
      const float rk = cop::ScanContext::ringKeyDist(Q.desc, M.desc);
      if (rk <= ringkey_gate_) cand.emplace_back(rk, m);
    }
    if (cand.empty()) return;
    const int K = std::min(top_k_, int(cand.size()));
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end());

    int best_m = -1, best_shift = 0; float best_sc = 1e9f;
    for (int i = 0; i < K; ++i)
    {
      int sh = 0;
      const float d = cop::ScanContext::distance(Q.desc, db_[cand[i].second].desc, sh);
      if (d < best_sc) { best_sc = d; best_m = cand[i].second; best_shift = sh; }
    }
    if (best_m < 0 || best_sc > sc_thresh_) return;
    const KF & M = db_[best_m];
    const double L = Q.path - M.path;

    // 2. VERIFY: GICP query -> submap(match), two inits, keep the better overlap
    CloudT::Ptr target = submap(best_m);
    pcl::KdTreeFLANN<PointT> kd;
    kd.setInputCloud(target);

    const Eigen::Isometry3d T_odom = M.pose.inverse() * Q.pose;       // odometry's T_mq
    Eigen::Matrix4f init_sc = Eigen::Matrix4f::Identity();
    init_sc.block<3, 3>(0, 0) =
      Eigen::AngleAxisf(scd_.yawFromShift(best_shift), Eigen::Vector3f::UnitZ()).toRotationMatrix();
    const Eigen::Matrix4f init_odom = T_odom.matrix().cast<float>();

    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setMaxCorrespondenceDistance(icp_max_corr_);
    gicp.setMaximumIterations(icp_iters_);
    gicp.setInputSource(Q.cloud);
    gicp.setInputTarget(target);

    bool ok = false; Eigen::Matrix4f T_best; double ov_best = -1.0, rmse_best = 1e9;
    for (const Eigen::Matrix4f & init : {init_odom, init_sc})
    {
      CloudT aligned;
      gicp.align(aligned, init);
      if (!gicp.hasConverged()) continue;
      double ov, rmse;
      overlap(aligned, kd, ov, rmse);
      if (ov > ov_best) { ov_best = ov; rmse_best = rmse; T_best = gicp.getFinalTransformation(); ok = true; }
    }
    if (!ok) { reject(qi, best_m, "gicp not converged"); return; }
    if (ov_best < min_overlap_) { reject(qi, best_m, "low overlap", ov_best, min_overlap_); return; }
    if (rmse_best > max_rmse_) { reject(qi, best_m, "high rmse", rmse_best, max_rmse_); return; }

    Eigen::Isometry3d Z(T_best.cast<double>());
    Z.linear() = Eigen::Quaterniond(Z.rotation()).normalized().toRotationMatrix();
    if (Z.translation().norm() > max_loop_trans_)
    { reject(qi, best_m, "loop translation", Z.translation().norm(), max_loop_trans_); return; }

    // 3. DRIFT GATE: registration vs odometry, budget grows with travelled path
    const Eigen::Isometry3d D = T_odom.inverse() * Z;
    const double e_t = D.translation().norm();
    const double e_r = Eigen::AngleAxisd(D.rotation()).angle();
    if (e_t > drift_ratio_ * L + drift_floor_)
    { reject(qi, best_m, "drift-implausible translation", e_t, drift_ratio_ * L + drift_floor_); return; }
    if (e_r > drift_rot_ratio_ * L + drift_rot_floor_)
    { reject(qi, best_m, "drift-implausible rotation", e_r, drift_rot_ratio_ * L + drift_rot_floor_); return; }

    // 4. OBSERVABILITY of the final alignment
    Loop lp;
    int degen_t = 0, degen_r = 0;
    if (!observability(Q.cloud, target, kd, Z, rmse_best, lp.info, lp.Pobs, degen_t, degen_r))
    { reject(qi, best_m, "too few planar correspondences"); return; }
    if (degen_t > max_degen_trans_ || degen_r > max_degen_rot_)
    { reject(qi, best_m, "degenerate registration (trans/rot axes)", degen_t, degen_r); return; }

    lp.q = qi; lp.m = best_m; lp.id = next_id_++;
    lp.Z = Z; lp.sc_dist = best_sc; lp.overlap = ov_best; lp.rmse = rmse_best; lp.degen_axes = degen_t;
    addVerified(lp);
    if (loop_debug_)
      RCLCPP_INFO(get_logger(),
        "VERIFIED loop#%u kf%d<->kf%d sc=%.3f overlap=%.2f rmse=%.3f degen_t=%d drift=%.2fm/%.0fm",
        lp.id, qi, best_m, best_sc, ov_best, rmse_best, degen_t, e_t, L);

    // 5. PCM + publish / retract
    runPCM();
    publishConstraints();
  }

  void overlap(const CloudT & aligned, pcl::KdTreeFLANN<PointT> & kd, double & ov, double & rmse) const
  {
    std::vector<int> idx(1); std::vector<float> d2(1);
    int inl = 0; double se = 0.0;
    const float thr2 = float(inlier_dist_ * inlier_dist_);
    for (const auto & p : aligned.points)
      if (kd.nearestKSearch(p, 1, idx, d2) > 0 && d2[0] <= thr2) { ++inl; se += d2[0]; }
    ov = aligned.empty() ? 0.0 : double(inl) / aligned.size();
    rmse = inl ? std::sqrt(se / inl) : 1e9;
  }

  // Point-to-plane information of T_mq = Z under a RIGHT perturbation (query frame,
  // GTSAM Pose3 tangent order rot,trans). Normalised exactly like the LIO's
  // /lio/localizability so the thresholds mean the same thing. Output information is
  // sigma-scaled along observable eigen-directions and ~0 along degenerate ones.
  bool observability(const CloudT::Ptr & src, const CloudT::Ptr & tgt, pcl::KdTreeFLANN<PointT> & kd,
                     const Eigen::Isometry3d & Z, double rmse, Mat6 & info, Eigen::Matrix3d & Pobs,
                     int & degen_t, int & degen_r) const
  {
    const Eigen::Matrix3d R = Z.rotation();
    Eigen::Matrix3d Ht = Eigen::Matrix3d::Zero(), Hr = Eigen::Matrix3d::Zero();
    std::vector<int> nn; std::vector<float> d2;
    const float thr2 = float(inlier_dist_ * inlier_dist_);
    double r2 = 0.0; int n = 0;
    for (const auto & ps : src->points)
    {
      const Eigen::Vector3d p(ps.x, ps.y, ps.z);
      PointT w; const Eigen::Vector3d pw = Z * p; w.x = pw.x(); w.y = pw.y(); w.z = pw.z();
      if (kd.nearestKSearch(w, 8, nn, d2) < 8 || d2[0] > thr2) continue;
      Eigen::Vector3d mu = Eigen::Vector3d::Zero();
      for (int k : nn) mu += Eigen::Vector3d(tgt->points[k].x, tgt->points[k].y, tgt->points[k].z);
      mu /= nn.size();
      Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
      for (int k : nn)
      {
        const Eigen::Vector3d v = Eigen::Vector3d(tgt->points[k].x, tgt->points[k].y, tgt->points[k].z) - mu;
        C += v * v.transpose();
      }
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
      const Eigen::Vector3d ev = es.eigenvalues();
      if (ev[0] > 0.1 * ev.sum()) continue;                         // not locally planar
      const Eigen::Vector3d nq = R.transpose() * es.eigenvectors().col(0);   // normal, query frame
      const Eigen::Vector3d a = p.cross(nq);
      Ht += nq * nq.transpose();
      Hr += a * a.transpose();
      r2 += p.squaredNorm();
      ++n;
    }
    if (n < 30) return false;
    Ht /= n;
    Hr /= n * std::max(r2 / n, 1e-6);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> est(Ht), esr(Hr);
    const double st = loop_sigma_xyz_ * (1.0 + rmse / std::max(inlier_dist_, 1e-3));
    const double sr = loop_sigma_rot_ * (1.0 + rmse / std::max(inlier_dist_, 1e-3));
    Eigen::Matrix3d It = Eigen::Matrix3d::Zero(), Ir = Eigen::Matrix3d::Zero();
    Pobs.setZero(); degen_t = degen_r = 0;
    for (int k = 0; k < 3; ++k)
    {
      const Eigen::Vector3d vt = est.eigenvectors().col(k), vr = esr.eigenvectors().col(k);
      const bool ok_t = est.eigenvalues()(k) >= degen_trans_thresh_;
      const bool ok_r = esr.eigenvalues()(k) >= degen_rot_thresh_;
      if (ok_t) Pobs += vt * vt.transpose(); else ++degen_t;
      if (!ok_r) ++degen_r;
      It += (ok_t ? 1.0 / (st * st) : 1e-6) * vt * vt.transpose();
      Ir += (ok_r ? 1.0 / (sr * sr) : 1e-6) * vr * vr.transpose();
    }
    info.setZero();
    info.block<3, 3>(0, 0) = Ir;
    info.block<3, 3>(3, 3) = It;
    return true;
  }

  // ---- PCM over loop transforms --------------------------------------------
  bool consistentOneWay(const Loop & a, const Loop & b) const
  {
    const Eigen::Isometry3d Oq = db_[a.q].pose.inverse() * db_[b.q].pose;
    const Eigen::Isometry3d Om = db_[a.m].pose.inverse() * db_[b.m].pose;
    const Eigen::Isometry3d E = (Om * b.Z).inverse() * (a.Z * Oq);     // identity if consistent
    const double Lsum = std::fabs(db_[a.q].path - db_[b.q].path) + std::fabs(db_[a.m].path - db_[b.m].path);
    const double et = (a.Pobs * E.translation()).norm();               // observable part only
    const double er = Eigen::AngleAxisd(E.rotation()).angle();
    return et <= pcm_trans_ratio_ * Lsum + pcm_trans_floor_ &&
           er <= pcm_rot_ratio_ * Lsum + pcm_rot_floor_;
  }

  void addVerified(const Loop & lp)
  {
    const int n = int(verified_.size());
    verified_.push_back(lp);
    for (auto & row : adj_) row.push_back(0);
    adj_.emplace_back(n + 1, 0);
    for (int i = 0; i < n; ++i)
    {
      const bool c = consistentOneWay(verified_[i], lp) && consistentOneWay(lp, verified_[i]);
      adj_[i][n] = adj_[n][i] = c ? 1 : 0;
    }
  }

  // greedy max-clique: seed at highest degree, add nodes adjacent to all in set
  void runPCM()
  {
    const int n = int(verified_.size());
    confirmed_.clear();
    if (n == 0) return;
    std::vector<int> deg(n, 0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) deg[i] += adj_[i][j];
    std::vector<int> order(n); for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return deg[a] > deg[b]; });
    std::vector<int> clique{order[0]};
    for (int k = 1; k < n; ++k)
    {
      const int v = order[k];
      bool all = true;
      for (int c : clique) if (!adj_[v][c]) { all = false; break; }
      if (all) clique.push_back(v);
    }
    if (int(clique.size()) >= pcm_min_clique_) confirmed_ = clique;
  }

  void publishConstraints()
  {
    std::vector<char> in(verified_.size(), 0);
    for (int ci : confirmed_) in[ci] = 1;
    for (size_t i = 0; i < verified_.size(); ++i)
    {
      Loop & lp = verified_[i];
      if (in[i] && !lp.published)
      {
        loop_pub_->publish(toMsg(lp, false));
        lp.published = true;
        ++n_published_;
      }
      else if (!in[i] && lp.published)
      {
        loop_pub_->publish(toMsg(lp, true));
        lp.published = false;
        if (loop_debug_) RCLCPP_INFO(get_logger(), "RETRACT loop#%u (left the PCM clique)", lp.id);
      }
    }
  }

  LoopMsg toMsg(const Loop & lp, bool retract) const
  {
    LoopMsg msg;
    msg.loop_id = lp.id;
    msg.retract = retract;
    msg.query_stamp = db_[lp.q].stamp;
    msg.match_stamp = db_[lp.m].stamp;
    const Eigen::Vector3d pq = db_[lp.q].pose.translation(), pm = db_[lp.m].pose.translation();
    msg.query_position.x = pq.x(); msg.query_position.y = pq.y(); msg.query_position.z = pq.z();
    msg.match_position.x = pm.x(); msg.match_position.y = pm.y(); msg.match_position.z = pm.z();
    msg.query_pose = toPoseMsg(db_[lp.q].pose);
    msg.match_pose = toPoseMsg(db_[lp.m].pose);
    msg.relative_pose = toPoseMsg(lp.Z);
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) msg.information[r * 6 + c] = lp.info(r, c);
    msg.degenerate_axes = uint8_t(lp.degen_axes);
    msg.overlap = lp.overlap;
    msg.fitness = lp.rmse * lp.rmse;
    msg.path_length = db_[lp.q].path - db_[lp.m].path;
    return msg;
  }

  void publishMarkers(const builtin_interfaces::msg::Time & stamp)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.header.frame_id = world_frame_;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    visualization_msgs::msg::Marker line;
    line.header.frame_id = world_frame_;
    line.header.stamp = stamp;
    line.ns = "loops"; line.id = 0;
    line.type = visualization_msgs::msg::Marker::LINE_LIST;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = 0.3;
    line.color.r = 0.1f; line.color.g = 1.0f; line.color.b = 0.2f; line.color.a = 0.9f;
    line.pose.orientation.w = 1.0;
    for (int ci : confirmed_)
    {
      geometry_msgs::msg::Point a, b;
      const Eigen::Vector3d pq = db_[verified_[ci].q].pose.translation();
      const Eigen::Vector3d pm = db_[verified_[ci].m].pose.translation();
      a.x = pq.x(); a.y = pq.y(); a.z = pq.z();
      b.x = pm.x(); b.y = pm.y(); b.z = pm.z();
      line.points.push_back(a); line.points.push_back(b);
    }
    if (!line.points.empty()) arr.markers.push_back(line);
    marker_pub_->publish(arr);

    if (loop_debug_ && confirmed_.size() != last_confirmed_)
    {
      RCLCPP_INFO(get_logger(), "loops: %zu verified, %zu PCM-confirmed, %zu rejected, %zu kf",
                  verified_.size(), confirmed_.size(), n_rejected_, db_.size());
      last_confirmed_ = confirmed_.size();
    }
  }

  // params
  std::string cloud_topic_, odom_topic_, env_topic_;
  std::string world_frame_ = "odom";   // frame of the incoming LIO clouds
  double kf_dist_, kf_time_, work_radius_, kf_min_move_;
  double struct_height_, min_struct_frac_;
  int min_struct_bins_;
  int top_k_;
  double sc_thresh_, ringkey_gate_, min_loop_path_;
  double icp_leaf_, icp_max_corr_, max_loop_trans_, inlier_dist_, min_overlap_, max_rmse_;
  int icp_max_pts_, icp_iters_, submap_kf_;
  double drift_ratio_, drift_floor_, drift_rot_ratio_, drift_rot_floor_;
  double degen_trans_thresh_, degen_rot_thresh_, loop_sigma_xyz_, loop_sigma_rot_;
  int max_degen_trans_, max_degen_rot_;
  double pcm_trans_ratio_, pcm_trans_floor_, pcm_rot_ratio_, pcm_rot_floor_;
  int pcm_min_clique_;
  bool skip_in_poor_, loop_debug_;

  // state
  cop::ScanContext scd_;
  std::deque<OdomSample> odom_buf_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> pending_;
  double path_total_ = 0.0;
  std::vector<KF> db_;
  std::vector<Loop> verified_;
  std::vector<std::vector<char>> adj_;
  std::vector<int> confirmed_;
  uint32_t next_id_ = 0;
  size_t n_rejected_ = 0, n_published_ = 0, last_confirmed_ = 9999;
  uint8_t env_class_ = EnvMsg::RICH;
  CloudT::Ptr accum_;
  Eigen::Vector3d last_kf_t_{0, 0, 0};
  bool kf_init_ = false;
  rclcpp::Time last_kf_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<EnvMsg>::SharedPtr env_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<LoopMsg>::SharedPtr loop_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LoopClosureNode>());
  rclcpp::shutdown();
  return 0;
}
