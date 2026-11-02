// =============================================================================
//  Stage B: LiDAR-inertial odometry (lio_node)     GNSS-denied multi-terrain loc.
// -----------------------------------------------------------------------------
//  Iterated error-state Kalman filter + incremental ikd-tree map, derived from
//  FAST-LIO2 (HKU-MARS, GPL-2.0, see LICENSE.FAST_LIO) with Copernicus additions:
//  degeneracy-aware update (solution remapping), gravity-referenced attitude
//  aiding, long static IMU init and gyro-saturation handling.
//
//  Subscribes : common.lid_topic   (sensor_msgs/PointCloud2, time-fixed by Stage A)
//               common.imu_topic   (sensor_msgs/Imu)
//  Publishes  : (internal pipeline topics — the robot-facing odometry, path and the
//               whole map -> odom -> base_frame TF come from backend_node)
//               /lio/odometry                   (IMU pose + twist in odom)
//               /lio/cloud_registered           (deskewed scan, odom frame)
//               /lio/cloud_body                 (deskewed scan, IMU frame)
//               /lio/cloud_effected, /lio/map, /lio/localizability
//  Service    : /lio/save_map                   (std_srvs/Trigger, needs pcd_save.pcd_save_en)
//
//  Frames: the filter runs in its own world frame W (the IMU pose at start-up; not
//  gravity-aligned). Every output is re-expressed in `odom` = gravity-levelled, with
//  origin and heading of base_frame at start-up. T_odom_W is constant once set, so
//  odom is continuous.
// =============================================================================
#include <omp.h>
#include <mutex>
#include <unordered_set>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "imu_processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "copernicus_localization/base_extrinsic.hpp"
#include "copernicus_localization/msg/localizability.hpp"
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)

using LocMsg = copernicus_localization::msg::Localizability;

double kdtree_incremental_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0;
int    add_point_size = 0, kdtree_delete_counter = 0;
bool   pcd_save_en = false, extrinsic_est_en = true;

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;

string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool   is_first_lidar = true;

/*** Degeneracy-aware update (solution remapping) ***/
bool   degen_en = true, degen_log = false, degen_soft = true;
double degen_trans_thresh = 0.01, degen_rot_thresh = 0.01;
M3D    degen_It(Eye3d), degen_Ir(Eye3d);           // normalized info, world frame
V3D    degen_t_eig(Zero3d), degen_r_eig(Zero3d);   // ascending
M3D    degen_t_vec(Eye3d), degen_r_vec(Eye3d);     // columns = eigvecs, world frame
uint8_t degen_mask = 0;
int    degen_num_eff = 0;

/*** Gravity-referenced attitude (roll/pitch) aiding ***/
// In an along-axis-degenerate scene (bridge deck, tunnel, runway) pitch is not
// observable through the IMU (tilt error leaks into along-axis acceleration, which
// the LiDAR cannot see) and the LiDAR only constrains attitude relative to its own
// map, so map and pose tilt together and z runs away. An absolute "up" direction in
// the body frame — the IMU's AHRS orientation (source 1) or the local ground-plane
// normal under a level-ground assumption (source 2) — is added as a measurement whose
// information is a fixed fraction (kappa) of the LiDAR's own roll/pitch information.
int    imu_init_samples = 10;
double gyro_saturation = 0.0, gyro_saturation_inflate = 100.0;
int    att_source = 0;                  // 0 off, 1 IMU orientation (AHRS), 2 ground plane
double att_kappa = 0.05, att_sigma_floor = 2.0 * M_PI / 180.0, att_max_angle = 30.0 * M_PI / 180.0;
double att_ground_rmin = 2.0, att_ground_rmax = 20.0;
bool   att_valid = false;
V3D    att_up_body(0, 0, 1);            // measured world-up in the IMU/body frame
double att_last_err = 0.0;
mutex  mtx_ahrs;
deque<pair<double, Eigen::Quaterniond>> ahrs_buffer;

/*** Output frames (REP-105) ***/
string odom_frame = "odom", base_frame = "base_link", imu_frame;   // imu_frame "" = IMU is base_frame
bool   odom_anchored = false;
Eigen::Isometry3d T_imu_base = Eigen::Isometry3d::Identity();   // base_frame pose in the IMU frame
Eigen::Isometry3d T_odom_W   = Eigen::Isometry3d::Identity();   // filter world -> odom (constant)
Eigen::Isometry3d T_odom_imu = Eigen::Isometry3d::Identity();   // IMU pose in odom, current scan

vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));

pcl::VoxelGrid<PointType> downSizeFilterSurf;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
        time_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    mtx_buffer.unlock();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "imu loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    if (att_source == 1 && msg->orientation_covariance[0] >= 0.0)
    {
        const auto &q = msg->orientation;
        const double qn = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        if (qn > 0.5)
        {
            lock_guard<mutex> lk(mtx_ahrs);
            ahrs_buffer.emplace_back(timestamp, Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized());
            while (ahrs_buffer.size() > 400) ahrs_buffer.pop_front();
        }
    }

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

/*** Degeneracy analysis + solution remapping (Zhang, Kaess & Singh, ICRA 2016).
 * The position columns of h_x are world-frame plane normals and the rotation
 * columns are body-frame (lever x normal). Their normalized information matrices
 * are eigen-decomposed separately; eigen-directions below threshold carry no real
 * LiDAR constraint (tunnel axis, runway x/y/yaw) yet thousands of tiny, over-
 * trusted residuals would still drag the state along them and collapse its
 * covariance. Projecting those directions out of h_x makes the iEKF leave both
 * the state and the covariance to the IMU prediction there. In well-constrained
 * scenes every projector is the identity, so the update is unchanged.
 * SOFT mode (default) scales a weak direction's information by lambda/thresh
 * instead of zeroing it: on PERSISTENT degeneracy (a runway, a long tunnel) a hard
 * projection leaves velocity unobserved and pure IMU integration diverges, while a
 * weak-but-real LiDAR constraint keeps it bounded. ***/
void remap_degenerate_directions(const state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    const int N = effct_feat_num;
    degen_num_eff = N;
    if (N < 6) return;

    double r2 = 0.0;
    for (int i = 0; i < N; i++)
    {
        const PointType &lp = laserCloudOri->points[i];
        r2 += (s.offset_R_L_I * V3D(lp.x, lp.y, lp.z) + s.offset_T_L_I).squaredNorm();
    }
    r2 = std::max(r2 / N, 1e-6);

    const Eigen::MatrixXd Ht = ekfom_data.h_x.leftCols(3);
    const Eigen::MatrixXd Hr = ekfom_data.h_x.middleCols(3, 3);
    const M3D It = (Ht.transpose() * Ht) / double(N);
    const M3D Ir_body = (Hr.transpose() * Hr) / (double(N) * r2);

    Eigen::SelfAdjointEigenSolver<M3D> est(It), esr(Ir_body);
    M3D Pt = M3D::Zero(), Pr = M3D::Zero();
    uint8_t mask = 0;
    for (int k = 0; k < 3; k++)
    {
        const V3D vt = est.eigenvectors().col(k), vr = esr.eigenvectors().col(k);
        const double lt = est.eigenvalues()(k), lr = esr.eigenvalues()(k);
        // row scale s -> information scale s^2 = lambda/thresh (soft) or 0 (hard)
        const double st = lt >= degen_trans_thresh ? 1.0 : (degen_soft ? std::sqrt(std::max(lt, 0.0) / degen_trans_thresh) : 0.0);
        const double sr = lr >= degen_rot_thresh ? 1.0 : (degen_soft ? std::sqrt(std::max(lr, 0.0) / degen_rot_thresh) : 0.0);
        Pt += st * vt * vt.transpose();
        Pr += sr * vr * vr.transpose();
        if (st < 1.0) mask |= uint8_t(1u << k);
        if (sr < 1.0) mask |= uint8_t(1u << (3 + k));
    }

    const M3D R = s.rot.toRotationMatrix();
    degen_It = It;
    degen_Ir = R * Ir_body * R.transpose();
    degen_t_eig = est.eigenvalues();
    degen_r_eig = esr.eigenvalues();
    degen_t_vec = est.eigenvectors();
    degen_r_vec = R * esr.eigenvectors();
    degen_mask = degen_en ? mask : 0;

    if (degen_en && mask)
    {
        ekfom_data.h_x.leftCols(3) = Ht * Pt;
        ekfom_data.h_x.middleCols(3, 3) = Hr * Pr;
    }
}

/*** Measured world-up in the body frame for this scan (before the update). ***/
void prepare_attitude_measurement(const state_ikfom &s)
{
    att_valid = false;
    if (att_source == 0) return;
    const V3D g(s.grav[0], s.grav[1], s.grav[2]);
    const V3D u_pred = s.rot.conjugate() * (-g.normalized());
    V3D u;
    if (att_source == 1)
    {
        lock_guard<mutex> lk(mtx_ahrs);
        double best = 0.05;
        bool found = false;
        for (const auto &e : ahrs_buffer)
            if (std::fabs(e.first - lidar_end_time) < best) { best = std::fabs(e.first - lidar_end_time); u = e.second.conjugate() * V3D(0, 0, 1); found = true; }
        if (!found) return;
    }
    else
    {
        // ground plane under the vehicle: lowest points (along predicted up) in an
        // annulus, then two PCA refits on the inliers
        vector<V3D> pts;
        pts.reserve(feats_down_body->size());
        for (const auto &p : feats_down_body->points)
        {
            const V3D pi = s.offset_R_L_I * V3D(p.x, p.y, p.z) + s.offset_T_L_I;
            const double r = (pi - pi.dot(u_pred) * u_pred).norm();
            if (r >= att_ground_rmin && r <= att_ground_rmax) pts.push_back(pi);
        }
        if (pts.size() < 100) return;
        vector<double> h(pts.size());
        for (size_t i = 0; i < pts.size(); i++) h[i] = pts[i].dot(u_pred);
        vector<double> hs = h;
        std::nth_element(hs.begin(), hs.begin() + hs.size() / 10, hs.end());
        const double h10 = hs[hs.size() / 10];
        vector<V3D> sel;
        for (size_t i = 0; i < pts.size(); i++) if (h[i] < h10 + 0.3) sel.push_back(pts[i]);
        V3D n = u_pred;
        for (int it = 0; it < 3 && sel.size() >= 50; it++)
        {
            V3D c = V3D::Zero();
            for (const auto &p : sel) c += p;
            c /= double(sel.size());
            M3D C = M3D::Zero();
            for (const auto &p : sel) C += (p - c) * (p - c).transpose();
            Eigen::SelfAdjointEigenSolver<M3D> es(C);
            n = es.eigenvectors().col(0);
            if (n.dot(u_pred) < 0) n = -n;
            vector<V3D> in;
            for (const auto &p : pts) if (std::fabs(n.dot(p - c)) < 0.1) in.push_back(p);
            sel.swap(in);
        }
        if (sel.size() < 100) return;
        u = n;
    }
    att_last_err = std::acos(std::min(1.0, std::max(-1.0, u.dot(u_pred))));
    if (att_last_err > att_max_angle) return;
    att_up_body = u;
    att_valid = true;
}

/*** Append the attitude rows: residual (u_meas - u_pred), Jacobian [u_pred]x on the
 * rotation block; weight = max(kappa * LiDAR roll/pitch information, 1/sigma_floor^2). ***/
void add_attitude_rows(const state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    if (!att_valid) return;
    const V3D g(s.grav[0], s.grav[1], s.grav[2]);
    const V3D u = s.rot.conjugate() * (-g.normalized());
    const M3D P_rp = M3D::Identity() - u * u.transpose();
    const Eigen::MatrixXd Hr = ekfom_data.h_x.middleCols(3, 3);
    const double lidar_rp = (P_rp * (Hr.transpose() * Hr)).trace() / 2.0;
    const double w = std::max(att_kappa * lidar_rp, LASER_POINT_COV / (att_sigma_floor * att_sigma_floor));
    const double sw = std::sqrt(w);
    M3D ux;
    ux << SKEW_SYM_MATRX(u);
    const int n0 = ekfom_data.h_x.rows();
    ekfom_data.h_x.conservativeResize(n0 + 3, Eigen::NoChange);
    ekfom_data.h.conservativeResize(n0 + 3);
    ekfom_data.h_x.bottomRows(3).setZero();
    ekfom_data.h_x.block(n0, 3, 3, 3) = sw * ux;
    ekfom_data.h.tail(3) = sw * (att_up_body - u);
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    remap_degenerate_directions(s, ekfom_data);
    add_attitude_rows(s, ekfom_data);
    solve_time += omp_get_wtime() - solve_start_;
}

/*** ---------------------------- outputs (odom frame) ---------------------------- ***/
using Mat6 = Eigen::Matrix<double, 6, 6>;

string sensor_frame() { return imu_frame.empty() ? base_frame : imu_frame; }

Eigen::Isometry3d imu_pose_world()
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = state_point.rot.toRotationMatrix();
    T.translation() = V3D(state_point.pos);
    return T;
}

/*** Fix odom on the first published scan: origin + heading of base_frame, z = -gravity. ***/
void anchor_odom()
{
    const Eigen::Isometry3d T_W_base = imu_pose_world() * T_imu_base;
    const V3D up = -V3D(state_point.grav[0], state_point.grav[1], state_point.grav[2]).normalized();
    V3D x = T_W_base.linear().col(0);
    x -= x.dot(up) * up;
    if (x.norm() < 1e-3) x = T_W_base.linear().col(1).cross(up);   // base x-axis vertical
    x.normalize();
    Eigen::Isometry3d T_W_odom = Eigen::Isometry3d::Identity();
    T_W_odom.linear() << x, up.cross(x), up;
    T_W_odom.translation() = T_W_base.translation();
    T_odom_W = T_W_odom.inverse();
    odom_anchored = true;
}

void pointLidarToOdom(PointType const * const pi, PointType * const po)
{
    const V3D p_imu(state_point.offset_R_L_I * V3D(pi->x, pi->y, pi->z) + state_point.offset_T_L_I);
    const V3D p = T_odom_imu * p_imu;
    po->x = p(0);
    po->y = p(1);
    po->z = p(2);
    po->intensity = pi->intensity;
}

template<typename T>
void fill_pose(const Eigen::Isometry3d &T_pose, T &out)
{
    const Eigen::Quaterniond q(T_pose.linear());
    out.position.x = T_pose.translation()(0);
    out.position.y = T_pose.translation()(1);
    out.position.z = T_pose.translation()(2);
    out.orientation.x = q.x();
    out.orientation.y = q.y();
    out.orientation.z = q.z();
    out.orientation.w = q.w();
}

void fill_odom_msg(nav_msgs::msg::Odometry &o, const string &child, const Eigen::Isometry3d &T_pose,
                   const V3D &v, const V3D &w, const Mat6 &C)
{
    o.header.frame_id = odom_frame;
    o.header.stamp = get_ros_time(lidar_end_time);
    o.child_frame_id = child;
    fill_pose(T_pose, o.pose.pose);
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++) o.pose.covariance[r * 6 + c] = C(r, c);
    o.twist.twist.linear.x = v(0);  o.twist.twist.linear.y = v(1);  o.twist.twist.linear.z = v(2);
    o.twist.twist.angular.x = w(0); o.twist.twist.angular.y = w(1); o.twist.twist.angular.z = w(2);
}

/*** IMU pose in odom for the pipeline + backend. Twist is in the IMU frame; pose
 * covariance is (x y z rx ry rz) in odom (REP-103 order). ***/
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pub)
{
    const M3D R_oW = T_odom_W.linear();
    const M3D R_oi = T_odom_imu.linear();

    // filter covariance: position error in W, rotation error in the IMU frame
    const auto P = kf.get_P();
    Mat6 J = Mat6::Zero();
    J.block<3, 3>(0, 0) = R_oW;
    J.block<3, 3>(3, 3) = R_oi;

    nav_msgs::msg::Odometry o;
    fill_odom_msg(o, sensor_frame(), T_odom_imu, state_point.rot.conjugate() * V3D(state_point.vel),
                  p_imu->angvel(), J * P.block<6, 6>(0, 0) * J.transpose());
    pub->publish(o);
}

void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
    PointCloudXYZI::Ptr cloud(dense_pub_en ? feats_undistort : feats_down_body);
    const int size = cloud->points.size();
    PointCloudXYZI::Ptr cloud_odom(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) pointLidarToOdom(&cloud->points[i], &cloud_odom->points[i]);

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud_odom, msg);
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = odom_frame;
    pub->publish(msg);
}

void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
    const int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr cloud_imu(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) RGBpointBodyLidarToIMU(&feats_undistort->points[i], &cloud_imu->points[i]);

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud_imu, msg);
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = sensor_frame();
    pub->publish(msg);
}

void publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
    PointCloudXYZI::Ptr cloud_odom(new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++) pointLidarToOdom(&laserCloudOri->points[i], &cloud_odom->points[i]);

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud_odom, msg);
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = odom_frame;
    pub->publish(msg);
}

/*** Accumulated map (one scan per publish period), also what /lio/save_map writes.
 * One point per map_voxel cell, so it grows with the area covered, not with run time. ***/
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
std::unordered_set<int64_t> map_voxels;
double map_voxel = 0.2;
void publish_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub)
{
    PointCloudXYZI::Ptr cloud(dense_pub_en ? feats_undistort : feats_down_body);
    const double inv = 1.0 / map_voxel;
    for (const auto &pt : cloud->points)
    {
        PointType po;
        pointLidarToOdom(&pt, &po);
        // 21 bits per axis (+-1M cells): 200 km at 0.1 m voxels
        const int64_t kx = int64_t(std::floor(po.x * inv)) & 0x1FFFFF;
        const int64_t ky = int64_t(std::floor(po.y * inv)) & 0x1FFFFF;
        const int64_t kz = int64_t(std::floor(po.z * inv)) & 0x1FFFFF;
        if (map_voxels.insert((kx << 42) | (ky << 21) | kz).second) pcl_wait_pub->push_back(po);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*pcl_wait_pub, msg);
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = odom_frame;
    pub->publish(msg);
}

void publish_localizability(const rclcpp::Publisher<LocMsg>::SharedPtr &pub)
{
    const M3D R_oW = T_odom_W.linear();
    const M3D It = R_oW * degen_It * R_oW.transpose();
    const M3D Ir = R_oW * degen_Ir * R_oW.transpose();
    const M3D Vt = R_oW * degen_t_vec, Vr = R_oW * degen_r_vec;

    LocMsg m;
    m.header.frame_id = odom_frame;
    m.header.stamp = get_ros_time(lidar_end_time);
    m.num_effective = degen_num_eff;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
        {
            m.trans_info[r * 3 + c] = It(r, c);
            m.rot_info[r * 3 + c] = Ir(r, c);
        }
    for (int k = 0; k < 3; k++)
    {
        m.trans_eigvals[k] = degen_t_eig(k);
        m.rot_eigvals[k] = degen_r_eig(k);
        m.trans_axes[k].x = Vt(0, k); m.trans_axes[k].y = Vt(1, k); m.trans_axes[k].z = Vt(2, k);
        m.rot_axes[k].x = Vr(0, k);   m.rot_axes[k].y = Vr(1, k);   m.rot_axes[k].z = Vr(2, k);
    }
    m.degenerate_mask = degen_mask;
    m.remapping_active = degen_en && degen_mask != 0;
    pub->publish(m);
}

class LioNode : public rclcpp::Node
{
public:
    LioNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("lio", options)
    {
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<double>("publish.map_voxel", 0.2);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/velodyne_points");
        this->declare_parameter<string>("common.imu_topic", "/imu/data");
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", VELO16);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter<bool>("degeneracy.enable", true);
        this->declare_parameter<double>("degeneracy.trans_thresh", 0.01);
        this->declare_parameter<double>("degeneracy.rot_thresh", 0.01);
        this->declare_parameter<bool>("degeneracy.log", false);
        this->declare_parameter<bool>("degeneracy.soft", true);
        this->declare_parameter<double>("mapping.gyro_saturation", 0.0);
        this->declare_parameter<double>("mapping.gyro_saturation_inflate", 100.0);
        this->declare_parameter<int>("mapping.imu_init_samples", 10);
        this->declare_parameter<int>("attitude.source", 0);
        this->declare_parameter<double>("attitude.kappa", 0.05);
        this->declare_parameter<double>("attitude.sigma_floor_deg", 2.0);
        this->declare_parameter<double>("attitude.max_angle_deg", 30.0);
        this->declare_parameter<double>("attitude.ground_min_range", 2.0);
        this->declare_parameter<double>("attitude.ground_max_range", 20.0);
        // odom anchor: origin + heading of base_frame at start-up (see base_extrinsic.hpp
        // for how base_frame <- imu_frame is resolved). backend_node reads the same block.
        this->declare_parameter<string>("frames.odom_frame", "odom");
        this->declare_parameter<string>("frames.base_frame", "base_link");
        this->declare_parameter<string>("frames.imu_frame", "");
        this->declare_parameter<vector<double>>("frames.base_to_imu", vector<double>());
        this->declare_parameter<double>("frames.tf_timeout", 5.0);

        double sig_deg = 2.0, max_deg = 30.0;
        this->get_parameter_or<bool>("degeneracy.soft", degen_soft, true);
        this->get_parameter_or<double>("mapping.gyro_saturation", gyro_saturation, 0.0);
        this->get_parameter_or<double>("mapping.gyro_saturation_inflate", gyro_saturation_inflate, 100.0);
        this->get_parameter_or<int>("mapping.imu_init_samples", imu_init_samples, 10);
        this->get_parameter_or<int>("attitude.source", att_source, 0);
        this->get_parameter_or<double>("attitude.kappa", att_kappa, 0.05);
        this->get_parameter_or<double>("attitude.sigma_floor_deg", sig_deg, 2.0);
        this->get_parameter_or<double>("attitude.max_angle_deg", max_deg, 30.0);
        this->get_parameter_or<double>("attitude.ground_min_range", att_ground_rmin, 2.0);
        this->get_parameter_or<double>("attitude.ground_max_range", att_ground_rmax, 20.0);
        att_sigma_floor = sig_deg * M_PI / 180.0;
        att_max_angle = max_deg * M_PI / 180.0;
        this->get_parameter_or<bool>("degeneracy.enable", degen_en, true);
        this->get_parameter_or<double>("degeneracy.trans_thresh", degen_trans_thresh, 0.01);
        this->get_parameter_or<double>("degeneracy.rot_thresh", degen_rot_thresh, 0.01);
        this->get_parameter_or<bool>("degeneracy.log", degen_log, false);

        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en_, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en_, false);
        this->get_parameter_or<double>("publish.map_voxel", map_voxel, 0.2);
        map_voxel = std::max(map_voxel, 0.01);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/velodyne_points");
        this->get_parameter_or<string>("common.imu_topic", imu_topic, "/imu/data");
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);
        this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);
        this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);
        this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);
        this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);
        this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, VELO16);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<string>("frames.odom_frame", odom_frame, "odom");
        this->get_parameter_or<string>("frames.base_frame", base_frame, "base_link");
        this->get_parameter_or<string>("frames.imu_frame", imu_frame, "");
        vector<double> base_to_imu;
        double tf_timeout = 5.0;
        this->get_parameter_or<vector<double>>("frames.base_to_imu", base_to_imu, vector<double>());
        this->get_parameter_or<double>("frames.tf_timeout", tf_timeout, 5.0);
        base_extrinsic_.configure(base_frame, imu_frame, base_to_imu, tf_timeout);
        imu_frame = base_extrinsic_.imuFrame();

        RCLCPP_INFO(this->get_logger(), "lidar_type %d, lid_topic: %s, imu_topic: %s",
                    p_pre->lidar_type, lid_topic.c_str(), imu_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "odom anchor frame: %s (IMU frame: %s)", base_frame.c_str(),
                    imu_frame.empty() ? "= base" : imu_frame.c_str());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_init_samples(imu_init_samples);
        p_imu->set_gyro_saturation(gyro_saturation, gyro_saturation_inflate);
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi_, epsi_ + 23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi_);

        /*** ROS interfaces ***/
        sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, rclcpp::SensorDataQoS(), imu_cbk);
        pubOdom_ = this->create_publisher<nav_msgs::msg::Odometry>("/lio/odometry", 20);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lio/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lio/cloud_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lio/cloud_effected", 20);
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/lio/map", 20);
        pubLocalizability_ = this->create_publisher<LocMsg>("/lio/localizability", 20);
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        RCLCPP_INFO(this->get_logger(), "degeneracy remapping: %s%s (trans_thresh=%.4f rot_thresh=%.4f)",
                    degen_en ? "on" : "off", degen_soft ? " soft" : " hard", degen_trans_thresh, degen_rot_thresh);
        RCLCPP_INFO(this->get_logger(), "attitude aiding: %s (kappa=%.3f floor=%.1fdeg)",
                    att_source == 1 ? "IMU orientation" : (att_source == 2 ? "ground plane" : "off"),
                    att_kappa, att_sigma_floor * 180.0 / M_PI);

        timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(10),
                                      std::bind(&LioNode::timer_callback, this));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), std::chrono::milliseconds(1000),
                                              std::bind(&LioNode::map_publish_callback, this));
        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("/lio/save_map",
            std::bind(&LioNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

private:
    void timer_callback()
    {
        if (!sync_packages(Measures)) return;

        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
            return;
        }

        match_time = 0;
        solve_time = 0;

        p_imu->Process(Measures, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return;
        }

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;
        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        /*** initialize the map kdtree ***/
        if (ikdtree.Root_Node == nullptr)
        {
            RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
            if (feats_down_size > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
            }
            return;
        }

        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return;
        }

        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);

        /*** iterated state estimation ***/
        double solve_H_time = 0;
        prepare_attitude_measurement(kf.get_x());
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        /*** add the feature points to map kdtree ***/
        map_incremental();

        if (degen_log && degen_mask)
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "degenerate: mask=0x%02x trans_eig=[%.4f %.4f %.4f] rot_eig=[%.4f %.4f %.4f] N=%d",
                degen_mask, degen_t_eig(0), degen_t_eig(1), degen_t_eig(2),
                degen_r_eig(0), degen_r_eig(1), degen_r_eig(2), degen_num_eff);

        /******* Publish (odom frame) *******/
        if (!base_extrinsic_.resolve(*this, *tf_buffer_, lidar_end_time)) return;
        if (!odom_anchored)
        {
            T_imu_base = base_extrinsic_.T_imu_base();
            anchor_odom();
            RCLCPP_INFO(this->get_logger(), "%s anchored at the start pose of %s (gravity-levelled)",
                        odom_frame.c_str(), base_frame.c_str());
        }
        T_odom_imu = T_odom_W * imu_pose_world();
        have_output_ = true;

        publish_odometry(pubOdom_);
        publish_localizability(pubLocalizability_);
        if (scan_pub_en)                     publish_frame_world(pubLaserCloudFull_);
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
        if (effect_pub_en_)                  publish_effect_world(pubLaserCloudEffect_);
    }

    void map_publish_callback()
    {
        if (map_pub_en_ && have_output_) publish_map(pubLaserCloudMap_);
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::ConstSharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        if (!pcd_save_en || map_file_path.empty() || pcl_wait_pub->empty())
        {
            res->success = false;
            res->message = !pcd_save_en ? "Map save disabled (pcd_save.pcd_save_en)."
                         : map_file_path.empty() ? "map_file_path not set." : "Map is empty (publish.map_en off?).";
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
        res->success = true;
        res->message = "Map saved to " + map_file_path;
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
    rclcpp::Publisher<LocMsg>::SharedPtr pubLocalizability_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr map_pub_timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en_ = false, map_pub_en_ = false, have_output_ = false;
    copernicus_localization::BaseExtrinsic base_extrinsic_;
    double epsi_[23] = {0.001};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LioNode>());
    rclcpp::shutdown();
    return 0;
}
