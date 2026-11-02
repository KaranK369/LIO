// =============================================================================
//  base_frame pose in the IMU frame (T_imu_base), shared by lio_node (odom anchor)
//  and backend_node (robot pose output). Source, in order:
//    imu_frame "" or == base_frame  -> identity (the IMU is the base)
//    base_to_imu [x y z roll pitch yaw] (imu_frame in base_frame) -> fixed value
//    TF base_frame <- imu_frame (robot_state_publisher / bag /tf_static)
//  If TF is still missing after `timeout` seconds, falls back to identity (loudly).
// =============================================================================
#pragma once

#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>

namespace copernicus_localization
{

class BaseExtrinsic
{
public:
  void configure(const std::string & base_frame, const std::string & imu_frame,
                 const std::vector<double> & base_to_imu, double timeout)
  {
    base_ = base_frame;
    imu_ = imu_frame == base_frame ? std::string() : imu_frame;
    base_to_imu_ = base_to_imu;
    timeout_ = timeout;
  }

  const std::string & imuFrame() const { return imu_; }
  bool ready() const { return ready_; }
  const Eigen::Isometry3d & T_imu_base() const { return T_imu_base_; }

  // true once T_imu_base is known; `now_sec` (data time) drives the TF timeout
  bool resolve(rclcpp::Node & node, const tf2_ros::Buffer & tf, double now_sec)
  {
    if (ready_) return true;
    if (imu_.empty()) return finish(Eigen::Isometry3d::Identity());
    if (base_to_imu_.size() == 6)
    {
      Eigen::Isometry3d T_base_imu = Eigen::Isometry3d::Identity();
      T_base_imu.translation() = Eigen::Vector3d(base_to_imu_[0], base_to_imu_[1], base_to_imu_[2]);
      T_base_imu.linear() = (Eigen::AngleAxisd(base_to_imu_[5], Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(base_to_imu_[4], Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(base_to_imu_[3], Eigen::Vector3d::UnitX())).toRotationMatrix();
      RCLCPP_INFO(node.get_logger(), "%s <- %s from frames.base_to_imu", base_.c_str(), imu_.c_str());
      return finish(T_base_imu.inverse());
    }
    try
    {
      const auto tr = tf.lookupTransform(imu_, base_, tf2::TimePointZero);
      const auto & t = tr.transform.translation;
      const auto & q = tr.transform.rotation;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.translation() = Eigen::Vector3d(t.x, t.y, t.z);
      T.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
      RCLCPP_INFO(node.get_logger(), "%s in %s from TF: t=[%.3f %.3f %.3f]",
                  base_.c_str(), imu_.c_str(), t.x, t.y, t.z);
      return finish(T);
    }
    catch (const tf2::TransformException & ex)
    {
      if (wait_start_ < 0.0) wait_start_ = now_sec;
      if (now_sec - wait_start_ < timeout_)
      {
        RCLCPP_WARN_THROTTLE(node.get_logger(), *node.get_clock(), 2000,
          "waiting for TF %s -> %s (outputs held): %s", base_.c_str(), imu_.c_str(), ex.what());
        return false;
      }
      RCLCPP_ERROR(node.get_logger(),
        "no TF %s -> %s after %.1fs: assuming IMU == %s. Set frames.base_to_imu or publish the robot TF.",
        base_.c_str(), imu_.c_str(), timeout_, base_.c_str());
      return finish(Eigen::Isometry3d::Identity());
    }
  }

private:
  bool finish(const Eigen::Isometry3d & T)
  {
    T_imu_base_ = T;
    ready_ = true;
    return true;
  }

  std::string base_, imu_;
  std::vector<double> base_to_imu_;
  double timeout_ = 5.0, wait_start_ = -1.0;
  bool ready_ = false;
  Eigen::Isometry3d T_imu_base_ = Eigen::Isometry3d::Identity();
};

}  // namespace copernicus_localization
