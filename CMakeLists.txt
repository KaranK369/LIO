cmake_minimum_required(VERSION 3.8)
project(copernicus_localization)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -DNDEBUG)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common kdtree search segmentation filters sample_consensus registration)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)

# ---- Interface messages -----------------------------------------------------
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Localizability.msg"
  "msg/LoopConstraint.msg"
  "msg/LandmarkObs.msg"
  "msg/EnvironmentState.msg"
  DEPENDENCIES geometry_msgs builtin_interfaces std_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)

# Install launch files and config files (including .osm and rviz configs)
install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME}/
)

# Stage B LiDAR-inertial odometry (iEKF + ikd-tree map, derived from FAST-LIO2):
#   publishes odom->base TF, /localization/odometry and the /lio/* pipeline topics.
add_executable(lio_node
  src/lio/lio_node.cpp
  src/lio/preprocess.cpp
  src/lio/ikd_Tree.cpp)
# include/ holds ikd-Tree/ and IKFoM_toolkit/ (third-party, header-only + ikd_Tree.cpp)
target_include_directories(lio_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/lio
  ${PCL_INCLUDE_DIRS})
# kNN/residual search threads in the iEKF update (FAST-LIO's MP_EN switch)
include(ProcessorCount)
ProcessorCount(N_CPU)
if(N_CPU GREATER 4)
  target_compile_definitions(lio_node PRIVATE MP_EN MP_PROC_NUM=3)
elseif(N_CPU GREATER 3)
  target_compile_definitions(lio_node PRIVATE MP_EN MP_PROC_NUM=2)
else()
  target_compile_definitions(lio_node PRIVATE MP_PROC_NUM=1)
endif()
# third-party estimator code: keep the build log readable
target_compile_options(lio_node PRIVATE -Wno-unused-parameter -Wno-unused-variable
  -Wno-unused-but-set-variable -Wno-sign-compare -Wno-pedantic -Wno-reorder
  -Wno-maybe-uninitialized -Wno-deprecated-copy -Wno-class-memaccess -Wno-parentheses)
ament_target_dependencies(lio_node
  rclcpp sensor_msgs nav_msgs geometry_msgs std_srvs tf2_ros pcl_conversions)
target_link_libraries(lio_node ${PCL_LIBRARIES} Eigen3::Eigen OpenMP::OpenMP_CXX ${cpp_typesupport_target})

# Stage A LiDAR preprocessing for the LIO:
#   time-fix + range crop + intensity/multi-return + Patchwork++-style ground
#   segmentation + lower-trunk band + rigidity + Dynablox-style dynamic removal.
add_executable(cloud_preprocess_node src/cloud_preprocess_node.cpp)
ament_target_dependencies(cloud_preprocess_node rclcpp sensor_msgs nav_msgs pcl_conversions)
target_include_directories(cloud_preprocess_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cloud_preprocess_node ${PCL_LIBRARIES} Eigen3::Eigen ${cpp_typesupport_target})

# Stage C geometric primitive landmarks (poles + planes), keyframe rate:
#   2D-footprint pole clustering + RANSAC vertical planes -> persistent
#   constellation map (see include/.../landmark_map.hpp). Publishes RViz markers.
add_executable(landmark_node src/landmark_node.cpp)
ament_target_dependencies(landmark_node
  rclcpp sensor_msgs nav_msgs visualization_msgs geometry_msgs pcl_conversions)
target_include_directories(landmark_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(landmark_node ${PCL_LIBRARIES} Eigen3::Eigen ${cpp_typesupport_target})

# Stage D loop-closure detector (keyframe rate): Scan Context candidate ->
#   GICP verification -> PCM consistency. Publishes loop-edge markers AND
#   /loop_constraints (LoopConstraint msg) for the Stage-E backend.
add_executable(loop_closure_node src/loop_closure_node.cpp)
ament_target_dependencies(loop_closure_node
  rclcpp sensor_msgs nav_msgs visualization_msgs geometry_msgs pcl_conversions)
target_include_directories(loop_closure_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(loop_closure_node ${PCL_LIBRARIES} Eigen3::Eigen ${cpp_typesupport_target})

# Environment router (§7): cheap keyframe-rate scene classification + degeneracy
#   (voxel-PCA dimensionality + point-to-plane/line localizability eigen-analysis).
#   Publishes /environment/state (EnvironmentState) for the backend to route on.
add_executable(environment_router_node src/environment_router_node.cpp)
ament_target_dependencies(environment_router_node
  rclcpp sensor_msgs nav_msgs visualization_msgs geometry_msgs std_msgs pcl_conversions)
target_include_directories(environment_router_node PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(environment_router_node ${PCL_LIBRARIES} Eigen3::Eigen ${cpp_typesupport_target})

set(COP_EXECUTABLES lio_node cloud_preprocess_node landmark_node loop_closure_node environment_router_node)

# ---- Stage E backend (GTSAM iSAM2) -----------------------------------------
# Guarded: only built when GTSAM is available (install `ros-humble-gtsam`).
# Keeps the package building without GTSAM; the backend appears once it is present.
find_package(GTSAM QUIET)
if(GTSAM_FOUND)
  message(STATUS "GTSAM ${GTSAM_VERSION} found - building Stage E backend_node")
  find_package(tf2 REQUIRED)
  find_package(tf2_geometry_msgs REQUIRED)
  add_executable(backend_node src/backend_node.cpp)
  ament_target_dependencies(backend_node
    rclcpp nav_msgs sensor_msgs geometry_msgs visualization_msgs std_msgs tf2 tf2_ros tf2_geometry_msgs)
  target_include_directories(backend_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${GTSAM_INCLUDE_DIR})
  target_link_libraries(backend_node Eigen3::Eigen gtsam ${cpp_typesupport_target})
  list(APPEND COP_EXECUTABLES backend_node)
else()
  message(WARNING
    "GTSAM not found - Stage E backend_node will NOT be built. "
    "Install it (e.g. `sudo apt install ros-humble-gtsam`) and rebuild.")
endif()

install(TARGETS ${COP_EXECUTABLES} DESTINATION lib/${PROJECT_NAME})

# Python nodes (OSM road map-matching, kept as a standalone utility)
install(PROGRAMS
  src/map_matching_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
