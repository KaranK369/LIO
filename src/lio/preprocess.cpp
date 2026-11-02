#include "preprocess.h"

#include <cmath>
#include <vector>

void Preprocess::process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (time_unit)
  {
    case SEC: time_unit_scale = 1.e3f; break;
    case MS:  time_unit_scale = 1.f; break;
    case US:  time_unit_scale = 1.e-3f; break;
    case NS:  time_unit_scale = 1.e-6f; break;
    default:  time_unit_scale = 1.f; break;
  }

  switch (lidar_type)
  {
    case OUST64:
      oust64_handler(*msg);
      break;
    case HESAI:
      hesai_handler(*msg);
      break;
    case VELO16:
    {
      // generic spinning LiDAR: pick the handler from the per-point time field
      bool has_timestamp = false, has_t = false;
      for (const auto &field : msg->fields)
      {
        if (field.name == "timestamp") { has_timestamp = true; break; }
        if (field.name == "t") { has_t = true; break; }
      }
      if (has_timestamp)  hesai_handler(*msg);
      else if (has_t)     oust64_handler(*msg);
      else                velodyne_handler(*msg);
      break;
    }
    default:
      RCLCPP_ERROR_ONCE(rclcpp::get_logger("lio_preprocess"),
                        "unsupported preprocess.lidar_type %d (use 2=Velodyne/Hesai, 3=Ouster, 5=Hesai)",
                        lidar_type);
      pl_surf.clear();
      break;
  }
  *pcl_out = pl_surf;
}

void Preprocess::oust64_handler(const sensor_msgs::msg::PointCloud2 &msg)
{
  pl_surf.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(msg, pl_orig);
  pl_surf.reserve(pl_orig.size());

  for (size_t i = 0; i < pl_orig.points.size(); i++)
  {
    if (i % point_filter_num != 0) continue;
    const auto &p = pl_orig.points[i];
    if (p.x * p.x + p.y * p.y + p.z * p.z < blind * blind) continue;

    PointType added_pt;
    added_pt.x = p.x;
    added_pt.y = p.y;
    added_pt.z = p.z;
    added_pt.intensity = p.intensity;
    added_pt.normal_x = added_pt.normal_y = added_pt.normal_z = 0;
    added_pt.curvature = p.t * time_unit_scale;  // ms
    pl_surf.points.push_back(added_pt);
  }
}

void Preprocess::hesai_handler(const sensor_msgs::msg::PointCloud2 &msg)
{
  pl_surf.clear();
  pcl::PointCloud<hesai_ros::Point> pl_orig;
  pcl::fromROSMsg(msg, pl_orig);
  if (pl_orig.points.empty()) return;
  pl_surf.reserve(pl_orig.size());

  // Hesai stamps are absolute seconds: offset relative to the first point
  const double time_beg = pl_orig.points[0].timestamp;
  for (size_t i = 0; i < pl_orig.points.size(); i++)
  {
    if (i % point_filter_num != 0) continue;
    const auto &p = pl_orig.points[i];
    if (p.x * p.x + p.y * p.y + p.z * p.z < blind * blind) continue;

    PointType added_pt;
    added_pt.x = p.x;
    added_pt.y = p.y;
    added_pt.z = p.z;
    added_pt.intensity = p.intensity;
    added_pt.normal_x = added_pt.normal_y = added_pt.normal_z = 0;
    const double dt = (p.timestamp - time_beg) * 1000.0;
    added_pt.curvature = dt > 0.0 ? dt : 0.0;  // ms
    pl_surf.points.push_back(added_pt);
  }
}

void Preprocess::velodyne_handler(const sensor_msgs::msg::PointCloud2 &msg)
{
  pl_surf.clear();
  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(msg, pl_orig);
  const size_t plsize = pl_orig.points.size();
  if (plsize == 0) return;
  pl_surf.reserve(plsize);

  // Per-point time from the `time` field if present, otherwise reconstructed from
  // the azimuth of each ring relative to its first point.
  const bool given_offset_time = pl_orig.points[plsize - 1].time > 0;
  const double omega_l = 0.361 * SCAN_RATE;  // scan angular velocity [deg/ms]
  std::vector<bool> is_first(N_SCANS, true);
  std::vector<double> yaw_fp(N_SCANS, 0.0);   // yaw of the first point of each ring
  std::vector<float> time_last(N_SCANS, 0.0); // last offset time of each ring

  for (size_t i = 0; i < plsize; i++)
  {
    const auto &p = pl_orig.points[i];
    PointType added_pt;
    added_pt.x = p.x;
    added_pt.y = p.y;
    added_pt.z = p.z;
    added_pt.intensity = p.intensity;
    added_pt.normal_x = added_pt.normal_y = added_pt.normal_z = 0;
    added_pt.curvature = p.time * time_unit_scale;  // ms

    if (!given_offset_time)
    {
      const int layer = p.ring;
      if (layer >= N_SCANS) continue;
      const double yaw_angle = std::atan2(added_pt.y, added_pt.x) * 57.2957;
      if (is_first[layer])
      {
        yaw_fp[layer] = yaw_angle;
        is_first[layer] = false;
        time_last[layer] = 0.0;
        continue;
      }
      added_pt.curvature = yaw_angle <= yaw_fp[layer]
        ? (yaw_fp[layer] - yaw_angle) / omega_l
        : (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
      if (added_pt.curvature < time_last[layer]) added_pt.curvature += 360.0 / omega_l;
      time_last[layer] = added_pt.curvature;
    }

    if (i % point_filter_num == 0 &&
        added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > blind * blind)
      pl_surf.points.push_back(added_pt);
  }
}
