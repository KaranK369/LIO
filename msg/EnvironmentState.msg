# Per-keyframe scene classification + degeneracy from the environment router (§7,
# environment_router_node). Consumed by the Stage-E backend to ROUTE estimator
# behavior: in degenerate/feature-poor scenes the LIO between-factor covariance is
# inflated ANISOTROPICALLY along the weak (poorly-constrained) directions, and loops
# are down-weighted in POOR. The router has no camera — it reasons purely from LiDAR
# geometry (point-to-plane / point-to-line localizability, Zhang-2016 style).
std_msgs/Header header

# scene tier (§7)
uint8 RICH=0      # structured (forest/urban): full primitive SLAM + loops trusted
uint8 SPARSE=1    # partial structure: opportunistic loops, scaled trust
uint8 POOR=2      # feature-poor / directionally degenerate: inflate cov, distrust loops
uint8 environment_class

# 0 = well-constrained in all directions .. 1 = a fully-unconstrained (degenerate)
# axis. Derived from the WEAKEST normalized localizability eigenvalue.
float32 degeneracy

# dimensionality mix of the scan (voxel fractions; sum ~ 1 over classified voxels)
float32 plane_ratio
float32 pole_ratio
float32 scatter_ratio
uint32  num_voxels        # structured voxels used for the estimate

# Directional localizability (WORLD frame), ascending order:
#   loc_axes[0] / loc_eigvals[0] = the WEAKEST-constrained direction.
# loc_eigvals[k] is PER-AXIS LOCALIZABILITY in [0,1] (1 = fully constrained, 0 =
# free/degenerate), an ABSOLUTE-reference measure (Zhang-2016 style). The backend
# inflates the between-factor translation covariance along each axis by a function
# of (1 - loc_eigvals[k]).
geometry_msgs/Vector3[] loc_axes
float32[] loc_eigvals

# ---- v2: full 6-DOF localizability ------------------------------------------
# Rotation localizability (WORLD frame axes, ascending, [0,1]); a low value on the
# vertical axis = yaw is unconstrained (open plain, circular tunnel).
geometry_msgs/Vector3[] rot_axes
float32[] rot_eigvals
uint8 SOURCE_CLOUD=0                    # estimated from cloud voxel-PCA (fallback)
uint8 SOURCE_LIO=1                      # from FAST-LIO's measurement Jacobian (/localizability)
uint8 source
