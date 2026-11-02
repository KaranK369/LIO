// =============================================================================
//  Stage E: pose-graph backend (GTSAM iSAM2)     GNSS-denied multi-terrain loc.
// -----------------------------------------------------------------------------
//  Subscribes : /lio/odometry      (lio_node IMU pose in odom: keyframes + graph)
//               /lio/localizability (Localizability: per-scan 6-DOF degeneracy)
//               /loop_constraints  (copernicus_localization/LoopConstraint, Stage D)
//               /landmark_obs      (copernicus_localization/LandmarkObs, Stage C)
//               /environment/state (copernicus_localization/EnvironmentState, §7)
//               /imu/mag           (optional magnetometer heading)
//  THE localization output (the only odometry / path / TF of the stack):
//  Publishes  : /backend/odometry  (drift-corrected base_frame pose + twist in map, full rate)
//               /backend/path      (optimised keyframe trajectory of base_frame, map)
//               TF  map -> odom          (the drift correction; jumps only on corrections)
//               TF  odom -> base_frame   (the LIO motion; smooth)
//  enable_graph:=false runs it as a pass-through (correction = identity, map == odom)
//  so the output interface is the same with or without the pose graph.
//
//  ONE robust incremental path — no modes, plug-and-play, built for long outdoor runs:
//    * iSAM2 incremental pose graph (relinearises only what a loop touches; the whole
//      trajectory stays globally consistent — no fixed-lag skeleton, no marginalisation,
//      so no dangling-key crashes and no frozen-window kinks).
//    * PriorFactor on keyframe 0 (anchor).
//    * BetweenFactor between consecutive keyframes from LIO relative motion
//      (the "LIO between-factor"): sigma grows with distance/rotation, and is inflated
//      ANISOTROPICALLY (6-DOF, incl. yaw) along the directions the LIO itself reported
//      as weakly constrained over that keyframe interval (/lio/localizability; router
//      state as fallback). Gain is capped so the chain never turns to jelly.
//    * Loop closures: associated to OUR keyframes by STAMP with exact odometry offset
//      compensation; noise = the detector's ANISOTROPIC information (≈0 along the
//      registration's degenerate axes, so a tunnel loop never snaps the tunnel axis);
//      a χ² gate against the current estimate + drift budget BEFORE insertion; Huber
//      for convex absorption of large legitimate loops; a post-optimisation residual
//      check, and RETRACTION (factor removal) when the detector withdraws a loop.
//    NOTE: with no loop / landmark / heading factor the optimum IS the odometry —
//    inflating LIO covariance cannot remove front-end drift by itself; the degenerate
//    directions are handled in lio_node (solution remapping).
//    * Pose<->landmark factors from Stage-C poles for continuous anchoring (optional).
//    * Optional magnetometer yaw factor (off by default — needs a calibrated mag).
//
//  Memory/compute: only the pose graph + landmark positions are kept (never clouds),
//  so long runs stay light; iSAM2 updates are incremental.
//
//  The drift correction is applied to EVERY incoming odom (not just keyframes):
//    corrected = T_correct * raw,   T_correct = est_latest * raw_latest^-1
//  so /backend/odometry and the map->odom TF are full rate. T_correct is a map<-odom
//  transform, so it applies unchanged to the IMU pose (graph) and the base pose (output).
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/expressions.h>
#include <gtsam/nonlinear/ExpressionFactor.h>

#include "copernicus_localization/msg/loop_constraint.hpp"
#include "copernicus_localization/msg/landmark_obs.hpp"
#include "copernicus_localization/msg/environment_state.hpp"
#include "copernicus_localization/msg/localizability.hpp"
#include "copernicus_localization/base_extrinsic.hpp"

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::L;
using EnvMsg = copernicus_localization::msg::EnvironmentState;
using LocMsg = copernicus_localization::msg::Localizability;

class BackendNode : public rclcpp::Node
{
public:
  BackendNode() : Node("backend_node")
  {
    kf_dist_ = declare_parameter<double>("kf_dist", 1.0);
    kf_time_ = declare_parameter<double>("kf_time", 2.0);
    // REP-105 TF for navigation: map->odom (the correction) + odom->base_frame (the LIO
    // motion), which Nav2 consumes (global_frame=map, robot_base_frame=base_frame).
    // odom_frame / base_frame / imu_frame MUST match the LIO config `frames:` block
    // (the launch file passes them through).
    enable_graph_ = declare_parameter<bool>("enable_graph", true);
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    map_frame_  = declare_parameter<std::string>("map_frame", "map");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    base_extrinsic_.configure(base_frame_,
      declare_parameter<std::string>("imu_frame", ""),
      declare_parameter<std::vector<double>>("base_to_imu", std::vector<double>{}),
      declare_parameter<double>("tf_timeout", 5.0));
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/lio/odometry");
    odom_sigma_xyz_ = declare_parameter<double>("odom_sigma_xyz", 0.05);    // floor per keyframe
    odom_sigma_rot_ = declare_parameter<double>("odom_sigma_rot", 0.01);
    odom_trans_ratio_ = declare_parameter<double>("odom_trans_ratio", 0.02); // sigma per metre
    odom_rot_ratio_   = declare_parameter<double>("odom_rot_ratio", 0.02);   // sigma per radian
    loop_sigma_xyz_ = declare_parameter<double>("loop_sigma_xyz", 0.20);
    loop_sigma_rot_ = declare_parameter<double>("loop_sigma_rot", 0.05);
    assoc_pos_ = declare_parameter<double>("assoc_pos", 3.0);   // nearest-keyframe radius [m]
    // Pole-landmark factors are OFF by default: on real data they can corrupt the
    // (already excellent) LIO trajectory — noisy pole z + nearest-neighbour
    // association occasionally yank the optimisation (measured: 0.0 m backend-vs-LIO
    // offset OFF vs 2.9-4.3 m + teleports ON). The robust drift-bounding core is
    // LIO odom + loop closures; landmarks need robust association before they
    // help and are left as an opt-in (with the anisotropic+Huber model below).
    enable_landmarks_ = declare_parameter<bool>("enable_landmarks", false);
    // pole-landmark observation noise — ANISOTROPIC: a vertical pole localises you well
    // HORIZONTALLY (x,y) but gives no reliable ELEVATION, so z is left very loose;
    // otherwise noisy pole-z observations drag the whole trajectory up/down (the
    // "path floating above the vehicle" artifact). lm_sigma_z >> lm_sigma_xy.
    lm_sigma_xy_ = declare_parameter<double>("lm_sigma_xy", 0.30);
    lm_sigma_z_  = declare_parameter<double>("lm_sigma_z", 10.0);
    lm_robust_k_ = declare_parameter<double>("lm_robust_k", 0.5);  // Huber threshold [m]
    lm_assoc_ = declare_parameter<double>("lm_assoc", 0.6);     // landmark assoc radius [m]
    // magnetometer heading factor (§6: bounds yaw in feature-poor terrain). DEFAULT
    // OFF: needs a hard/soft-iron-calibrated mag; an uncalibrated mag corrupts yaw.
    enable_heading_ = declare_parameter<bool>("enable_heading", false);
    heading_sigma_ = declare_parameter<double>("heading_sigma", 0.15);
    mag_topic_ = declare_parameter<std::string>("mag_topic", "/imu/mag");
    // §7 environment router: anisotropic LIO-covariance inflation along weak axes when
    // the scene is degenerate + loop down-weight in POOR. No-op unless /environment/state
    // is received and reports degeneracy (fully back-compatible).
    enable_router_ = declare_parameter<bool>("enable_router", true);
    env_topic_ = declare_parameter<std::string>("env_topic", "/environment/state");
    loc_topic_ = declare_parameter<std::string>("localizability_topic", "/lio/localizability");
    router_inflate_gain_ = declare_parameter<double>("router_inflate_gain", 9.0);   // max 10x sigma
    router_deg_floor_ = declare_parameter<double>("router_deg_floor", 0.30);
    loc_ref_t_ = declare_parameter<double>("loc_ref_trans", 0.10);   // normalised eig >= -> no inflation
    loc_ref_r_ = declare_parameter<double>("loc_ref_rot", 0.10);
    loop_extra_iters_ = declare_parameter<int>("loop_extra_iters", 10);
    // loop acceptance: chi2 of the loop vs the current estimate, with the loop's own
    // covariance + a drift budget for the odometry chain between the two keyframes
    loop_chi2_ = declare_parameter<double>("loop_chi2", 22.46);      // 6 dof, p=0.999
    loop_post_chi2_ = declare_parameter<double>("loop_post_chi2", 100.0);   // ~10 sigma residual
    drift_ratio_ = declare_parameter<double>("drift_ratio", 0.05);
    drift_floor_ = declare_parameter<double>("drift_floor", 1.0);
    drift_rot_ratio_ = declare_parameter<double>("drift_rot_ratio", 0.005);
    drift_rot_floor_ = declare_parameter<double>("drift_rot_floor", 0.1);
    assoc_dt_ = declare_parameter<double>("assoc_max_dt", 2.5);      // loop<->keyframe stamp gap [s]
    env_axes_ = {Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()};
    env_eig_ = {1.0, 1.0, 1.0};   // per-axis localizability (1 = fully constrained)
    env_rot_axes_ = env_axes_;

    isam_params_.relinearizeThreshold = 0.01;
    isam_params_.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(isam_params_);

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos, std::bind(&BackendNode::odomCb, this, std::placeholders::_1));
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/backend/odometry", qos);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/backend/path", qos);
    tf_bc_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    if (!enable_graph_)
    {
      have_correct_ = true;   // identity correction: map == odom
      RCLCPP_INFO(get_logger(),
        "backend_node: pose graph OFF (pass-through, map == odom) -> /backend/odometry,/path,TF %s->%s->%s",
        map_frame_.c_str(), odom_frame_.c_str(), base_frame_.c_str());
      return;
    }
    loop_sub_ = create_subscription<copernicus_localization::msg::LoopConstraint>(
      "/loop_constraints", rclcpp::QoS(rclcpp::KeepLast(100)).reliable(),
      std::bind(&BackendNode::loopCb, this, std::placeholders::_1));
    if (enable_landmarks_)
      lm_sub_ = create_subscription<copernicus_localization::msg::LandmarkObs>(
        "/landmark_obs", rclcpp::QoS(rclcpp::KeepLast(100)).reliable(),
        std::bind(&BackendNode::lmObsCb, this, std::placeholders::_1));
    if (enable_heading_)
      mag_sub_ = create_subscription<sensor_msgs::msg::MagneticField>(
        mag_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
        std::bind(&BackendNode::magCb, this, std::placeholders::_1));
    if (enable_router_)
    {
      env_sub_ = create_subscription<EnvMsg>(
        env_topic_, rclcpp::QoS(rclcpp::KeepLast(20)).reliable(),
        std::bind(&BackendNode::envCb, this, std::placeholders::_1));
      loc_sub_ = create_subscription<LocMsg>(
        loc_topic_, rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
        [this](const LocMsg::ConstSharedPtr m) {
          loc_buf_.push_back(m);
          while (loc_buf_.size() > 300) loc_buf_.pop_front();
        });
    }
    RCLCPP_INFO(get_logger(),
      "backend_node: iSAM2 pose graph (LIO + Huber loops%s%s) -> /backend/odometry,/path,TF %s->%s->%s | kf=%.1fm/%.1fs",
      enable_landmarks_ ? " + landmarks" : "", enable_heading_ ? " + heading" : "",
      map_frame_.c_str(), odom_frame_.c_str(), base_frame_.c_str(), kf_dist_, kf_time_);
  }

private:
  // ---- small helpers --------------------------------------------------------
  static gtsam::Pose3 toPose3(const geometry_msgs::msg::Pose & p)
  {
    return gtsam::Pose3(
      gtsam::Rot3::Quaternion(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
      gtsam::Point3(p.position.x, p.position.y, p.position.z));
  }
  static double stampSec(const builtin_interfaces::msg::Time & t)
  { return double(t.sec) + double(t.nanosec) * 1e-9; }
  static double wrap(double a)
  { while (a > M_PI) a -= 2.0 * M_PI; while (a < -M_PI) a += 2.0 * M_PI; return a; }

  void magCb(const sensor_msgs::msg::MagneticField::ConstSharedPtr m)
  { mag_yaw_ = std::atan2(m->magnetic_field.y, m->magnetic_field.x); have_mag_ = true; }

  void envCb(const EnvMsg::ConstSharedPtr m)
  {
    env_class_ = m->environment_class;
    env_deg_ = m->degeneracy;
    for (int k = 0; k < 3 && k < int(m->loc_axes.size()); ++k)
    {
      env_axes_[k] = Eigen::Vector3d(m->loc_axes[k].x, m->loc_axes[k].y, m->loc_axes[k].z);
      env_eig_[k] = (k < int(m->loc_eigvals.size())) ? double(m->loc_eigvals[k]) : 1.0;
    }
    for (int k = 0; k < 3; ++k)
    {
      env_rot_axes_[k] = k < int(m->rot_axes.size())
        ? Eigen::Vector3d(m->rot_axes[k].x, m->rot_axes[k].y, m->rot_axes[k].z) : Eigen::Vector3d::Unit(k);
      env_rot_eig_[k] = k < int(m->rot_eigvals.size()) ? double(m->rot_eigvals[k]) : 1.0;
    }
    have_env_ = true;
  }

  // yaw-only Pose3 prior (only rot_z constrained)
  gtsam::Pose3 yawPose(const gtsam::Pose3 & p, double yaw) const
  { return gtsam::Pose3(gtsam::Rot3::Yaw(yaw), p.translation()); }
  gtsam::SharedNoiseModel headingNoise() const
  { return gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 1e6, 1e6, heading_sigma_, 1e6, 1e6, 1e6).finished()); }

  // Per-axis localizability (eigvecs as columns, WORLD frame; values in [0,1]) for
  // the keyframe interval (t0, t1]: the LIO's averaged normalised information if
  // available, else the router's latest state. Returns false if nothing is known.
  bool intervalLocalizability(double t0, double t1, Eigen::Matrix3d & Vt, Eigen::Vector3d & lt,
                              Eigen::Matrix3d & Vr, Eigen::Vector3d & lr) const
  {
    Eigen::Matrix3d It = Eigen::Matrix3d::Zero(), Ir = Eigen::Matrix3d::Zero();
    int n = 0;
    for (const auto & m : loc_buf_)
    {
      const double ts = stampSec(m->header.stamp);
      if (ts <= t0 - 0.05 || ts > t1 + 0.05 || m->num_effective < 6) continue;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) { It(r, c) += m->trans_info[r * 3 + c]; Ir(r, c) += m->rot_info[r * 3 + c]; }
      ++n;
    }
    if (n > 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> est(It / n), esr(Ir / n);
      Vt = est.eigenvectors(); Vr = esr.eigenvectors();
      for (int k = 0; k < 3; ++k)
      {
        lt[k] = std::clamp(est.eigenvalues()(k) / loc_ref_t_, 0.0, 1.0);
        lr[k] = std::clamp(esr.eigenvalues()(k) / loc_ref_r_, 0.0, 1.0);
      }
      return true;
    }
    if (!have_env_ || env_deg_ < router_deg_floor_) return false;
    for (int k = 0; k < 3; ++k)
    {
      Vt.col(k) = env_axes_[k]; lt[k] = env_eig_[k];
      Vr.col(k) = env_rot_axes_[k]; lr[k] = env_rot_eig_[k];
    }
    return true;
  }

  // Consecutive LIO between-factor noise. Base sigmas grow with the keyframe's
  // translation / rotation; along weakly-constrained directions (WORLD frame,
  // rotated into the from-pose body frame) the sigma is inflated by up to
  // (1 + router_inflate_gain), so the LIO is distrusted only where it cannot see.
  gtsam::SharedNoiseModel odomNoise(const gtsam::Pose3 & from_pose, const gtsam::Pose3 & rel, double t0, double t1)
  {
    const double sx = std::max(odom_sigma_xyz_, odom_trans_ratio_ * rel.translation().norm());
    const double sr = odom_sigma_rot_ + odom_rot_ratio_ * gtsam::Rot3::Logmap(rel.rotation()).norm();
    Eigen::Matrix3d Vt, Vr; Eigen::Vector3d lt, lr;
    if (!enable_router_ || !intervalLocalizability(t0, t1, Vt, lt, Vr, lr))
      return gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << sr, sr, sr, sx, sx, sx).finished());

    const Eigen::Matrix3d Rwb = from_pose.rotation().matrix();   // body -> world
    Eigen::Matrix3d Ct = Eigen::Matrix3d::Zero(), Cr = Eigen::Matrix3d::Zero();
    double worst = 1.0;
    for (int k = 0; k < 3; ++k)
    {
      const double st = sx * (1.0 + router_inflate_gain_ * (1.0 - lt[k]));
      const double srk = sr * (1.0 + router_inflate_gain_ * (1.0 - lr[k]));
      const Eigen::Vector3d at = Rwb.transpose() * Vt.col(k), ar = Rwb.transpose() * Vr.col(k);
      Ct += (st * st) * (at * at.transpose());
      Cr += (srk * srk) * (ar * ar.transpose());
      worst = std::min({worst, lt[k], lr[k]});
    }
    gtsam::Matrix6 C = gtsam::Matrix6::Zero();
    C.block<3, 3>(0, 0) = Cr;                 // Pose3 tangent: (rot, trans)
    C.block<3, 3>(3, 3) = Ct;
    if (worst < 0.99)
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
        "backend: weakly constrained interval -> odom cov inflated (loc_t=[%.2f %.2f %.2f] loc_r=[%.2f %.2f %.2f])",
        lt[0], lt[1], lt[2], lr[0], lr[1], lr[2]);
    return gtsam::noiseModel::Gaussian::Covariance(C);
  }

  // ---- main odom callback: keyframe gate + full-rate correction ------------
  void odomCb(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const gtsam::Pose3 raw = toPose3(msg->pose.pose);
    const double t = stampSec(msg->header.stamp);

    // drop non-finite odometry: a single NaN/Inf pose would feed a degenerate factor
    // into iSAM2 (IndeterminantLinearSystemException -> node abort). Long-run nav must
    // survive a corrupt upstream message, so reject it and keep the last good correction.
    if (!finite(raw) || !std::isfinite(t))
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "backend: dropping non-finite LIO odometry message");
      return;
    }

    const bool first = kf_raw_.empty();
    bool make_kf = first;
    if (!first)
    {
      const double moved = kf_raw_.back().between(raw).translation().norm();
      make_kf = moved >= kf_dist_ || (t - kf_times_.back()) >= kf_time_;
    }
    if (!base_extrinsic_.resolve(*this, *tf_buffer_, t)) return;   // outputs need base<-IMU
    if (make_kf)
    {
      if (enable_graph_) addKeyframe(raw, t);
      else { kf_raw_.push_back(raw); kf_times_.push_back(t); publishPath(); }
    }

    publishOutputs(*msg, raw);   // apply current correction to EVERY odom
  }

  // pose of keyframe i in map (IMU frame): optimised, or raw in pass-through
  gtsam::Pose3 kfPose(size_t i) const
  { return enable_graph_ ? est_.at<gtsam::Pose3>(X(int(i))) : kf_raw_[i]; }

  gtsam::Pose3 imuToBase() const
  {
    const Eigen::Isometry3d & T = base_extrinsic_.T_imu_base();
    return gtsam::Pose3(gtsam::Rot3(T.linear()), gtsam::Point3(T.translation()));
  }

  static void setPose(geometry_msgs::msg::Pose & out, const gtsam::Pose3 & p)
  {
    out.position.x = p.x(); out.position.y = p.y(); out.position.z = p.z();
    const auto q = p.rotation().toQuaternion();
    out.orientation.w = q.w(); out.orientation.x = q.x();
    out.orientation.y = q.y(); out.orientation.z = q.z();
  }

  // /backend/odometry (base_frame in map) + TF map->odom, odom->base_frame. `raw` is the
  // LIO IMU pose in odom; twist (IMU frame) and covariance (odom frame) come with it.
  void publishOutputs(const nav_msgs::msg::Odometry & lio, const gtsam::Pose3 & raw)
  {
    // map->odom: identity until the first correction so the TF tree is never broken
    // at startup; piecewise-constant between keyframes.
    const gtsam::Pose3 corr = have_correct_ ? T_correct_ : gtsam::Pose3();
    const gtsam::Pose3 odom_base = raw * imuToBase();
    const gtsam::Pose3 map_base = corr * odom_base;
    if (!finite(corr) || !finite(map_base))
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "backend: non-finite correction, skipping outputs this cycle");
      return;
    }

    // base twist (base frame) from the IMU twist: v_b = R_bi (v_i + w_i x t_ib)
    const Eigen::Isometry3d & T_ib = base_extrinsic_.T_imu_base();
    const Eigen::Matrix3d R_bi = T_ib.linear().transpose();
    const auto & tw = lio.twist.twist;
    const Eigen::Vector3d v_i(tw.linear.x, tw.linear.y, tw.linear.z);
    const Eigen::Vector3d w_i(tw.angular.x, tw.angular.y, tw.angular.z);
    const Eigen::Vector3d v_b = R_bi * (v_i + w_i.cross(T_ib.translation()));
    const Eigen::Vector3d w_b = R_bi * w_i;

    // LIO pose covariance (x y z rx ry rz, odom) -> base_frame (lever arm) -> map.
    // This is the front-end's local uncertainty, not the pose-graph marginal.
    using Mat6 = Eigen::Matrix<double, 6, 6>;
    Mat6 C_imu;
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c) C_imu(r, c) = lio.pose.covariance[r * 6 + c];
    const Eigen::Vector3d r = raw.rotation().matrix() * T_ib.translation();
    Mat6 J = Mat6::Identity();
    J.block<3, 3>(0, 3) << 0.0, r.z(), -r.y(), -r.z(), 0.0, r.x(), r.y(), -r.x(), 0.0;   // -[r]x
    const Eigen::Matrix3d Rc = corr.rotation().matrix();
    J.block<3, 3>(0, 0) = Rc * J.block<3, 3>(0, 0);
    J.block<3, 3>(0, 3) = Rc * J.block<3, 3>(0, 3);
    J.block<3, 3>(3, 3) = Rc;
    const Mat6 C = J * C_imu * J.transpose();

    nav_msgs::msg::Odometry o;
    o.header = lio.header;
    o.header.frame_id = map_frame_;
    o.child_frame_id = base_frame_;
    setPose(o.pose.pose, map_base);
    for (int i = 0; i < 6; ++i)
      for (int k = 0; k < 6; ++k) o.pose.covariance[i * 6 + k] = C(i, k);
    o.twist.twist.linear.x = v_b.x();  o.twist.twist.linear.y = v_b.y();  o.twist.twist.linear.z = v_b.z();
    o.twist.twist.angular.x = w_b.x(); o.twist.twist.angular.y = w_b.y(); o.twist.twist.angular.z = w_b.z();
    odom_pub_->publish(o);

    if (!publish_tf_) return;
    // skip duplicate / non-monotonic stamps -> avoids TF_REPEATED_DATA spam in tf2
    const double ts = stampSec(lio.header.stamp);
    if (have_tf_stamp_ && ts <= last_tf_stamp_) return;
    last_tf_stamp_ = ts; have_tf_stamp_ = true;

    std::vector<geometry_msgs::msg::TransformStamped> tfs(2);
    tfs[0].header = lio.header; tfs[0].header.frame_id = map_frame_;  tfs[0].child_frame_id = odom_frame_;
    tfs[1].header = lio.header; tfs[1].header.frame_id = odom_frame_; tfs[1].child_frame_id = base_frame_;
    const gtsam::Pose3 * poses[2] = {&corr, &odom_base};
    for (int k = 0; k < 2; ++k)
    {
      geometry_msgs::msg::Pose p;
      setPose(p, *poses[k]);
      tfs[k].transform.translation.x = p.position.x;
      tfs[k].transform.translation.y = p.position.y;
      tfs[k].transform.translation.z = p.position.z;
      tfs[k].transform.rotation = p.orientation;
    }
    tf_bc_->sendTransform(tfs);
  }

  void addKeyframe(const gtsam::Pose3 & raw, double t)
  {
    const int i = int(kf_raw_.size());
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    if (i == 0)
    {
      auto prior = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), raw, prior));
      initial.insert(X(0), raw);
    }
    else
    {
      const gtsam::Pose3 rel = kf_raw_.back().between(raw);     // T_prev_cur (odom)
      graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(i - 1), X(i), rel,
        odomNoise(kf_raw_.back(), rel, kf_times_.back(), t)));
      initial.insert(X(i), est_.at<gtsam::Pose3>(X(i - 1)) * rel);  // consistent init
    }

    // magnetometer heading factor (yaw-only prior, map frame)
    double hyaw = std::numeric_limits<double>::quiet_NaN();
    if (enable_heading_ && have_mag_)
    {
      if (!offset_set_) { heading_offset_ = wrap(mag_yaw_ - raw.rotation().yaw()); offset_set_ = true; }
      hyaw = wrap(mag_yaw_ - heading_offset_);
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(i), yawPose(raw, hyaw), headingNoise()));
    }

    isam_->update(graph, initial);
    est_ = isam_->calculateEstimate();

    kf_path_.push_back(kf_raw_.empty() ? 0.0
      : kf_path_.back() + (raw.translation() - kf_raw_.back().translation()).norm());
    kf_raw_.push_back(raw);
    kf_times_.push_back(t);
    kf_heading_.push_back(hyaw);
    updateCorrection();
    publishPath();
  }

  // ---- loop closure --------------------------------------------------------
  int nearestKfByTime(double t) const
  {
    if (kf_times_.empty()) return -1;
    const auto it = std::lower_bound(kf_times_.begin(), kf_times_.end(), t);
    int best = -1; double bd = assoc_dt_;
    for (auto j : {it - kf_times_.begin() - 1, it - kf_times_.begin()})
      if (j >= 0 && j < long(kf_times_.size()) && std::fabs(kf_times_[j] - t) < bd)
      { bd = std::fabs(kf_times_[j] - t); best = int(j); }
    return best;
  }

  void loopCb(const copernicus_localization::msg::LoopConstraint::ConstSharedPtr msg)
  {
    if (msg->retract)
    {
      const auto it = loop_factor_.find(msg->loop_id);
      if (it == loop_factor_.end()) return;
      removeLoopFactor(it->second);
      loop_factor_.erase(it);
      RCLCPP_WARN(get_logger(), "backend: loop#%u RETRACTED by detector", msg->loop_id);
      return;
    }

    // Associate by STAMP (unambiguous), then compensate exactly for the offset
    // between the detector's keyframes and ours using the raw odometry poses:
    //   T_bm_bq = (raw_bm^-1 * match_pose) * T_mq * (query_pose^-1 * raw_bq)
    const int qi = nearestKfByTime(stampSec(msg->query_stamp));
    const int mi = nearestKfByTime(stampSec(msg->match_stamp));
    if (qi < 0 || mi < 0 || qi == mi)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "backend: loop#%u skipped (no keyframe near its stamps; q=%d m=%d)", msg->loop_id, qi, mi);
      return;
    }
    const gtsam::Pose3 T_mq = toPose3(msg->relative_pose);
    const bool have_poses = msg->query_pose.orientation.w != 0.0 || msg->query_pose.orientation.x != 0.0 ||
                            msg->query_pose.orientation.y != 0.0 || msg->query_pose.orientation.z != 0.0;
    const gtsam::Pose3 rel = have_poses
      ? kf_raw_[mi].between(toPose3(msg->match_pose)) * T_mq * toPose3(msg->query_pose).between(kf_raw_[qi])
      : T_mq;

    // noise: the detector's anisotropic information (rot, trans; query frame)
    gtsam::Matrix6 info;
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) info(r, c) = msg->information[r * 6 + c];
    gtsam::Matrix6 Cloop;
    if (info.cwiseAbs().maxCoeff() > 0.0)
      Cloop = (info + 1e-9 * gtsam::Matrix6::Identity()).inverse();
    else
    {
      const double sf = std::max(0.05, std::sqrt(std::max(0.0, msg->fitness)));
      const double sr = loop_sigma_rot_, st = loop_sigma_xyz_ * sf;
      Cloop = (gtsam::Vector(6) << sr*sr, sr*sr, sr*sr, st*st, st*st, st*st).finished().asDiagonal();
    }

    // chi2 gate against the CURRENT estimate, allowing for odometry drift between
    // the two keyframes. Gross aliases (claiming two places 50 m apart coincide)
    // never enter the graph.
    const double L = std::fabs(kf_path_[qi] - kf_path_[mi]);
    const double st = drift_ratio_ * L + drift_floor_, srr = drift_rot_ratio_ * L + drift_rot_floor_;
    gtsam::Matrix6 Cchain = (gtsam::Vector(6) << srr*srr, srr*srr, srr*srr, st*st, st*st, st*st).finished().asDiagonal();
    const gtsam::Pose3 pred = est_.at<gtsam::Pose3>(X(mi)).between(est_.at<gtsam::Pose3>(X(qi)));
    const gtsam::Vector6 e = rel.localCoordinates(pred);
    const double d2 = e.dot((Cloop + Cchain).inverse() * e);
    if (!std::isfinite(d2) || d2 > loop_chi2_)
    {
      ++n_loops_rejected_;
      RCLCPP_WARN(get_logger(),
        "backend: loop#%u kf%d<->kf%d REJECTED by chi2 gate (d2=%.1f > %.1f, |e_t|=%.2fm over %.0fm path)",
        msg->loop_id, qi, mi, d2, loop_chi2_, e.tail<3>().norm(), L);
      return;
    }

    // Huber keeps the cost convex so a large LEGITIMATE loop (big accumulated drift)
    // still pulls the graph closed; outliers are stopped by the gates above and the
    // post-optimisation check below, not by a redescending kernel.
    auto base = gtsam::noiseModel::Gaussian::Covariance(Cloop);
    auto robust = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(1.345), base);
    gtsam::NonlinearFactorGraph g;
    g.add(gtsam::BetweenFactor<gtsam::Pose3>(X(mi), X(qi), rel, robust));
    const gtsam::ISAM2Result res = isam_->update(g, gtsam::Values());
    for (int k = 0; k < loop_extra_iters_; ++k) isam_->update();   // propagate the closure
    est_ = isam_->calculateEstimate();
    const size_t fidx = res.newFactorsIndices.empty() ? 0 : res.newFactorsIndices[0];

    // post-optimisation consistency: a loop that still fights the rest of the graph
    // after convergence is inconsistent with it -> remove it.
    const gtsam::Pose3 post = est_.at<gtsam::Pose3>(X(mi)).between(est_.at<gtsam::Pose3>(X(qi)));
    const gtsam::Vector6 ep = rel.localCoordinates(post);
    const double d2p = ep.dot(Cloop.inverse() * ep);
    if (!res.newFactorsIndices.empty() && d2p > loop_post_chi2_)
    {
      removeLoopFactor(fidx);
      ++n_loops_rejected_;
      RCLCPP_WARN(get_logger(), "backend: loop#%u kf%d<->kf%d REMOVED after optimisation (residual d2=%.1f)",
        msg->loop_id, qi, mi, d2p);
      return;
    }
    if (!res.newFactorsIndices.empty()) loop_factor_[msg->loop_id] = fidx;
    ++n_loops_;
    updateCorrection();
    publishPath();
    RCLCPP_INFO(get_logger(),
      "backend: loop#%u kf%d<->kf%d applied (d2=%.1f, degen_axes=%u, overlap=%.2f) | %d loops, %d rejected | %zu kf",
      msg->loop_id, qi, mi, d2, msg->degenerate_axes, msg->overlap, n_loops_, n_loops_rejected_, kf_raw_.size());
  }

  void removeLoopFactor(size_t idx)
  {
    gtsam::FactorIndices rm{idx};
    isam_->update(gtsam::NonlinearFactorGraph(), gtsam::Values(), rm);
    for (int k = 0; k < loop_extra_iters_; ++k) isam_->update();
    est_ = isam_->calculateEstimate();
    if (n_loops_ > 0) --n_loops_;
    updateCorrection();
    publishPath();
  }

  // ---- Stage-C pole landmarks: pose<->landmark factors, incremental --------
  void lmObsCb(const copernicus_localization::msg::LandmarkObs::ConstSharedPtr msg)
  {
    if (kf_raw_.empty()) return;
    const int i = nearestKfByPos(msg->keyframe_position);
    if (i < 0) return;
    // anisotropic (tight x,y / loose z) + Huber: poles pin horizontal position, never
    // elevation, and a stray association/observation is down-weighted, not trusted.
    auto lmbase = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << lm_sigma_xy_, lm_sigma_xy_, lm_sigma_z_).finished());
    const gtsam::SharedNoiseModel lmnoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(lm_robust_k_), lmbase);
    gtsam::NonlinearFactorGraph g;
    gtsam::Values v;
    int added = 0;
    for (const auto & pt : msg->positions)
    {
      const gtsam::Point3 w(pt.x, pt.y, pt.z);
      int j = nearestLm(w);
      if (j < 0) { j = int(lm_world_.size()); lm_world_.push_back(w); v.insert(L(j), w); }
      g.addExpressionFactor(lmnoise, kf_raw_[i].transformTo(w),
        gtsam::transformTo(gtsam::Pose3_(X(i)), gtsam::Point3_(L(j))));
      ++added;
    }
    if (added)
    {
      isam_->update(g, v);
      est_ = isam_->calculateEstimate();
      updateCorrection();
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
      "backend: %zu landmarks, %zu keyframes", lm_world_.size(), kf_raw_.size());
  }

  int nearestKfByPos(const geometry_msgs::msg::Point & p) const
  {
    const gtsam::Point3 q(p.x, p.y, p.z);
    int best = -1; double bd = assoc_pos_;
    for (size_t i = 0; i < kf_raw_.size(); ++i)
    { const double d = (kf_raw_[i].translation() - q).norm(); if (d < bd) { bd = d; best = int(i); } }
    return best;
  }
  int nearestLm(const gtsam::Point3 & w) const
  {
    int best = -1; double bd = lm_assoc_;
    for (size_t j = 0; j < lm_world_.size(); ++j)
    { const double d = (lm_world_[j] - w).norm(); if (d < bd) { bd = d; best = int(j); } }
    return best;
  }

  // T_correct maps a raw LIO (odom-frame) pose to the optimised map frame.
  void updateCorrection()
  {
    if (kf_raw_.empty()) return;
    const int last = int(kf_raw_.size()) - 1;
    T_correct_ = est_.at<gtsam::Pose3>(X(last)) * kf_raw_[last].inverse();
    have_correct_ = true;
  }

  static bool finite(const gtsam::Pose3 & p)
  {
    const auto t = p.translation();
    const auto q = p.rotation().toQuaternion();
    return std::isfinite(t.x()) && std::isfinite(t.y()) && std::isfinite(t.z()) &&
           std::isfinite(q.w()) && std::isfinite(q.x()) && std::isfinite(q.y()) && std::isfinite(q.z());
  }

  void publishPath()
  {
    nav_msgs::msg::Path path;
    path.header.frame_id = map_frame_;
    path.header.stamp = now();
    const gtsam::Pose3 imu_base = imuToBase();
    for (size_t i = 0; i < kf_raw_.size(); ++i)
    {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      setPose(ps.pose, kfPose(i) * imu_base);   // base_frame trajectory
      path.poses.push_back(ps);
    }
    path_pub_->publish(path);
  }

  // ---- params ---------------------------------------------------------------
  double kf_dist_, kf_time_, odom_sigma_xyz_, odom_sigma_rot_, loop_sigma_xyz_, loop_sigma_rot_, assoc_pos_;
  double odom_trans_ratio_, odom_rot_ratio_, loc_ref_t_, loc_ref_r_;
  double loop_chi2_, loop_post_chi2_, drift_ratio_, drift_floor_, drift_rot_ratio_, drift_rot_floor_, assoc_dt_;
  bool enable_graph_, publish_tf_, enable_landmarks_, enable_heading_, enable_router_;
  std::string map_frame_, odom_frame_, base_frame_, odom_topic_;
  copernicus_localization::BaseExtrinsic base_extrinsic_;
  double last_tf_stamp_ = 0.0; bool have_tf_stamp_ = false;
  double lm_sigma_xy_, lm_sigma_z_, lm_robust_k_, lm_assoc_, heading_sigma_;
  double router_inflate_gain_, router_deg_floor_;
  int loop_extra_iters_;
  std::string mag_topic_, env_topic_, loc_topic_;

  // ---- state ----------------------------------------------------------------
  std::unique_ptr<gtsam::ISAM2> isam_;
  gtsam::ISAM2Params isam_params_;
  gtsam::Values est_;
  std::vector<gtsam::Pose3> kf_raw_;       // raw LIO pose per keyframe
  std::vector<double> kf_times_;
  std::vector<double> kf_path_;            // cumulative raw odometry path length per keyframe
  std::vector<double> kf_heading_;         // per-kf mag yaw target (NaN = none)
  std::vector<gtsam::Point3> lm_world_;    // landmark world positions (for association)
  double mag_yaw_ = 0.0, heading_offset_ = 0.0;
  bool have_mag_ = false, offset_set_ = false;
  gtsam::Pose3 T_correct_;
  bool have_correct_ = false;
  int n_loops_ = 0, n_loops_rejected_ = 0;
  std::unordered_map<uint32_t, size_t> loop_factor_;   // loop_id -> iSAM2 factor index
  std::deque<LocMsg::ConstSharedPtr> loc_buf_;
  // §7 router state
  uint8_t env_class_ = EnvMsg::RICH;
  double env_deg_ = 0.0;
  std::array<Eigen::Vector3d, 3> env_axes_;
  std::array<double, 3> env_eig_;
  std::array<Eigen::Vector3d, 3> env_rot_axes_;
  std::array<double, 3> env_rot_eig_{1.0, 1.0, 1.0};
  bool have_env_ = false;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<copernicus_localization::msg::LoopConstraint>::SharedPtr loop_sub_;
  rclcpp::Subscription<copernicus_localization::msg::LandmarkObs>::SharedPtr lm_sub_;
  rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub_;
  rclcpp::Subscription<EnvMsg>::SharedPtr env_sub_;
  rclcpp::Subscription<LocMsg>::SharedPtr loc_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_bc_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BackendNode>());
  rclcpp::shutdown();
  return 0;
}
