# A verified, PCM-confirmed loop closure produced by Stage D (loop_closure_node)
# and consumed by the Stage E backend. Keyframes are identified by TIMESTAMP
# (not index) so the backend associates them to its own keyframe nodes by nearest
# time — decoupling the two nodes' independent keyframe bookkeeping.
builtin_interfaces/Time query_stamp     # the revisiting (newer) keyframe time
builtin_interfaces/Time match_stamp     # the matched (older) keyframe time
geometry_msgs/Point query_position      # odom-frame position of query keyframe (robust association)
geometry_msgs/Point match_position      # odom-frame position of match keyframe
geometry_msgs/Pose relative_pose        # T_match_query: query frame expressed in match frame (from GICP)
float64 fitness                         # GICP fitness (lower = better) -> backend noise model

# ---- v2: exact association + anisotropic trust + retraction ----------------
uint32 loop_id                          # unique per detector run
bool retract                            # true: withdraw a previously published loop_id
geometry_msgs/Pose query_pose           # raw FAST-LIO odom pose of the query keyframe
geometry_msgs/Pose match_pose           # raw FAST-LIO odom pose of the match keyframe
# 6x6 row-major information of relative_pose in GTSAM Pose3 tangent order
# (rot_x rot_y rot_z trans_x trans_y trans_z), expressed in the QUERY frame. Nearly
# zero along directions the registration could not observe (tunnel axis, plane-on-
# plane slide), so the backend never snaps the graph along them. All zeros = unset.
float64[36] information
uint8 degenerate_axes                   # number of translation axes judged degenerate
float64 overlap                         # inlier fraction of the verified registration
float64 path_length                     # odometry path length between match and query [m]
