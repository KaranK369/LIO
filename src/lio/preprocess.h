// LiDAR scan conversion for the LIO front-end (derived from FAST-LIO2, GPL-2.0).
// Converts a PointCloud2 into PointXYZINormal with the per-point time offset [ms]
// stored in `curvature`, applying point decimation and the blind-range crop.
// Only spinning LiDARs are supported (Velodyne, Ouster, Hesai).
#pragma once

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// Numbering kept from FAST-LIO so existing configs stay valid.
enum LID_TYPE
{
  VELO16 = 2,  // spinning; a `timestamp` field selects Hesai, a `t` field Ouster
  OUST64 = 3,
  HESAI = 5
};
enum TIME_UNIT
{
  SEC = 0,
  MS = 1,
  US = 2,
  NS = 3
};

namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (float, time, time)(uint16_t, ring, ring))

namespace ouster_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint16_t ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)
    (std::uint16_t, ring, ring)(std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range))

namespace hesai_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace hesai_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (double, timestamp, timestamp)(uint16_t, ring, ring))

class Preprocess
{
public:
  void process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out);

  int lidar_type = VELO16, point_filter_num = 1, N_SCANS = 16, SCAN_RATE = 10, time_unit = US;
  double blind = 0.01;

private:
  void oust64_handler(const sensor_msgs::msg::PointCloud2 &msg);
  void velodyne_handler(const sensor_msgs::msg::PointCloud2 &msg);
  void hesai_handler(const sensor_msgs::msg::PointCloud2 &msg);

  PointCloudXYZI pl_surf;
  float time_unit_scale = 1.f;
};
