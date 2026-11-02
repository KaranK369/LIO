# Per-keyframe geometric-landmark observations from Stage C (landmark_node),
# consumed by the Stage E backend to add pose<->landmark factors. The keyframe is
# referenced by POSITION so the backend associates it to its own keyframe node by
# nearest position (same scheme as LoopConstraint), decoupling the two nodes'
# independent keyframe bookkeeping.
builtin_interfaces/Time stamp
geometry_msgs/Point keyframe_position    # odom-frame position of the observing keyframe
geometry_msgs/Point[] positions          # WORLD-frame positions of poles observed this keyframe
