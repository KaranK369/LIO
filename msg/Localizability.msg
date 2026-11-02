# Per-scan localizability of the lio_node iEKF LiDAR update: eigen-analysis of
# the point-to-plane measurement Jacobian (Zhang, Kaess & Singh, ICRA 2016).
#
# trans_info = sum(n n^T) / N           (n = unit plane normal; trace == 1)
# rot_info   = sum(a a^T) / (N mean r^2) (a = lever x n; dimensionless)
# Both are in the odom frame (header.frame_id), row-major 3x3. Eigenvalues are ascending,
# so index 0 is the WEAKEST-constrained direction. An eigen-direction whose value
# is below the configured threshold is "degenerate": it was projected out of the
# update (solution remapping) and the IMU prediction carries the state along it.
std_msgs/Header header
int32 num_effective                # effective point-to-plane correspondences
float64[9] trans_info
float64[9] rot_info
float64[3] trans_eigvals
geometry_msgs/Vector3[3] trans_axes
float64[3] rot_eigvals
geometry_msgs/Vector3[3] rot_axes
uint8 degenerate_mask              # bit k (k=0..2) trans eig k, bit 3+k rot eig k
bool remapping_active              # true if the update was actually remapped
