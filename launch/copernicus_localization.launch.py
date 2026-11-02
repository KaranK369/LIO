#!/usr/bin/env python3
# =============================================================================
#  Copernicus GNSS-denied multi-terrain localization
#
#    <lid_topic> ─▶ cloud_preprocess (time-fix + scene-adaptive filtering)
#                       ─▶ <lid_topic>_processed  (+ IMU)
#                                 ▼
#                   lio_node (iEKF + ikd-tree map, degeneracy-aware)
#                       ─▶ /lio/odometry, /lio/cloud_registered, /lio/localizability (internal)
#                                 ▼
#        environment_router ─▶ /environment/state ─▶ preprocess / loop / backend
#        loop_closure_node  ─▶ /loop_constraints  ─▶ backend (iSAM2)
#                                 ▼
#  OUTPUT (backend_node only): /backend/odometry, /backend/path,
#                              TF map ─▶ odom ─▶ base_frame   (REP-105)
#  base_frame -> sensors is the robot's own TF (not launched here); base<-IMU is read
#  from it once (or from frames.base_to_imu in the LIO config).
#  enable_backend:=false keeps the same outputs with the pose graph off (map == odom).
#
#  Play the bag alongside this, e.g.:
#    ros2 bag play /home/karan_k/AIRL/bridge2 --clock
#
#  Args:
#    config_file   LIO config in config/ (or absolute path), e.g. lio_husky.yaml
#    platform      node-parameter profile: config/platform_<platform>.yaml
#                  (default derived from config_file, see PLATFORM_BY_CONFIG)
#    params_file   explicit node-parameter YAML (overrides `platform`)
#    enable_*      node on/off switches; the enable_filtering / denoise / rigidity /
#                  dynamic flags OVERRIDE the platform YAML only when given
#    use_sim_time  (default false)  rviz (default true)
#
#  All tuning lives in the platform YAML. Launch-time dicts override YAML in ROS 2,
#  so this file passes only topics, use_sim_time, the LiDAR extrinsic and the odom frame
#  (read from the LIO config, single source of truth) and switches you set explicitly.
# =============================================================================

from os.path import join, isabs, exists
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PLATFORM_BY_CONFIG = {
    'lio_bridge2.yaml': 'geode_vlp16',
    'lio_urban_tunnel.yaml': 'geode_vlp16',
    'lio_runwayD.yaml': 'runwayD',
    'lio_copernicus.yaml': 'copernicus',
    'lio_hesai.yaml': 'hilti_handheld',
    'lio_husky.yaml': 'husky',
    'lio_husky_garage.yaml': 'husky',
}

LIO_ODOM = '/lio/odometry'
LIO_CLOUD = '/lio/cloud_registered'

# launch arg -> cloud_preprocess parameter; applied only when the arg is non-empty
PREPROCESS_SWITCHES = {
    'enable_filtering': 'enable_filtering',
    'enable_denoise': 'enable_denoise',
    'enable_rigidity': 'enable_rigidity',
    'enable_dynamic': 'enable_dynamic_removal',
    'dynamic_debug': 'dynamic_debug',
}


def _as_bool(v):
    return str(v).strip().lower() in ('true', '1', 'yes', 'on')


def _launch_setup(context):
    pkg = get_package_share_directory('copernicus_localization')
    arg = lambda name: LaunchConfiguration(name).perform(context)

    cfg_name = arg('config_file')
    lio_cfg = cfg_name if isabs(cfg_name) else join(pkg, 'config', cfg_name)
    with open(lio_cfg, 'r') as f:
        ros_params = (yaml.safe_load(f) or {}).get('/**', {}).get('ros__parameters', {})
    lid_topic = ros_params.get('common', {}).get('lid_topic', '/velodyne_points')
    mapping = ros_params.get('mapping', {})
    frames = ros_params.get('frames', {})
    odom_frame = frames.get('odom_frame', 'odom')
    map_frame = 'map'

    params_file = arg('params_file')
    if not params_file:
        platform = arg('platform') or PLATFORM_BY_CONFIG.get(cfg_name.split('/')[-1], 'copernicus')
        params_file = join(pkg, 'config', f'platform_{platform}.yaml')
    if not exists(params_file):
        raise RuntimeError(f'node parameter file not found: {params_file}')

    use_sim_time = _as_bool(arg('use_sim_time'))
    preprocess_on = _as_bool(arg('enable_preprocess'))
    processed_topic = f'{lid_topic}_processed'
    common = {'use_sim_time': use_sim_time}

    # ── Stage A: time-fix (always required for Velodyne) + scene-adaptive filters
    pre_overrides = dict(common, input_topic=lid_topic, output_topic=processed_topic,
                         odom_topic=LIO_ODOM)
    if len(mapping.get('extrinsic_T', [])) == 3:
        pre_overrides['lidar_extrinsic_T'] = [float(v) for v in mapping['extrinsic_T']]
    if len(mapping.get('extrinsic_R', [])) == 9:
        pre_overrides['lidar_extrinsic_R'] = [float(v) for v in mapping['extrinsic_R']]
    for launch_arg, param in PREPROCESS_SWITCHES.items():
        v = arg(launch_arg)
        if v != '':
            pre_overrides[param] = _as_bool(v)

    nodes = [LogInfo(msg=f'[copernicus] LIO config:  {lio_cfg}'),
             LogInfo(msg=f'[copernicus] node params: {params_file}'),
             LogInfo(msg=f"[copernicus] TF: {map_frame} -> {odom_frame} -> {frames.get('base_frame', 'base_link')}")]
    if preprocess_on:
        nodes.append(Node(
            package='copernicus_localization', executable='cloud_preprocess_node',
            name='cloud_preprocess', output='screen',
            parameters=[params_file, pre_overrides]))

    # ── Stage B: LIO front-end (reads the time-fixed cloud)
    lio_overrides = {
        'use_sim_time': use_sim_time,
        'common.lid_topic': processed_topic if preprocess_on else lid_topic,
    }
    # publish_map:=false drops the heavy visualization outputs (/lio/map,
    # /lio/cloud_effected, body-frame scan); /lio/cloud_registered stays on
    # because the router and loop closure consume it. '' = use the YAML.
    if arg('publish_map') != '':
        on = _as_bool(arg('publish_map'))
        lio_overrides.update({'publish.map_en': on, 'publish.effect_map_en': on,
                              'publish.scan_bodyframe_pub_en': on})
    nodes.append(Node(
        package='copernicus_localization', executable='lio_node', name='lio', output='screen',
        parameters=[lio_cfg, lio_overrides]))

    # ── Stage C: geometric primitive landmarks (poles + planes)
    if _as_bool(arg('enable_landmarks')):
        nodes.append(Node(
            package='copernicus_localization', executable='landmark_node',
            name='landmark_node', output='screen',
            parameters=[params_file, dict(common, cloud_topic=LIO_CLOUD, odom_topic=LIO_ODOM)]))

    # ── Stage D: loop-closure detector
    if _as_bool(arg('enable_loop')):
        nodes.append(Node(
            package='copernicus_localization', executable='loop_closure_node',
            name='loop_closure_node', output='screen',
            parameters=[params_file, dict(common, cloud_topic=LIO_CLOUD, odom_topic=LIO_ODOM,
                                          loop_debug=_as_bool(arg('loop_debug')))]))

    # ── §7: environment router
    router_on = _as_bool(arg('enable_router'))
    if router_on:
        nodes.append(Node(
            package='copernicus_localization', executable='environment_router_node',
            name='environment_router_node', output='screen',
            parameters=[params_file, dict(common, cloud_topic=LIO_CLOUD, odom_topic=LIO_ODOM)]))

    # ── Stage E: GTSAM iSAM2 pose-graph backend = the single localization output
    #    (/backend/odometry, /backend/path, TF map->odom->base_frame). Always runs;
    #    enable_backend:=false only switches the pose graph off (pass-through).
    backend_overrides = dict(common, enable_graph=_as_bool(arg('enable_backend')),
                             enable_router=router_on, enable_heading=_as_bool(arg('enable_heading')),
                             odom_topic=LIO_ODOM, map_frame=map_frame, odom_frame=odom_frame,
                             base_frame=frames.get('base_frame', 'base_link'),
                             imu_frame=frames.get('imu_frame', ''),
                             tf_timeout=float(frames.get('tf_timeout', 5.0)),
                             publish_tf=bool(frames.get('publish_tf', True)))
    if len(frames.get('base_to_imu', [])) == 6:   # empty lists can't be passed as parameters
        backend_overrides['base_to_imu'] = [float(v) for v in frames['base_to_imu']]
    nodes.append(Node(
        package='copernicus_localization', executable='backend_node',
        name='backend_node', output='screen',
        parameters=[params_file, backend_overrides]))
    return nodes


def generate_launch_description():
    pkg = get_package_share_directory('copernicus_localization')
    use_sim_time = LaunchConfiguration('use_sim_time')

    args = [
        DeclareLaunchArgument('config_file', default_value='lio_copernicus.yaml',
                              description='LIO config file in config/ or full path'),
        DeclareLaunchArgument('platform', default_value='',
                              description='platform_<name>.yaml profile (default: from config_file)'),
        DeclareLaunchArgument('params_file', default_value='',
                              description='explicit node-parameter YAML (overrides platform)'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('enable_preprocess', default_value='true',
                              description='false = the LIO reads the raw topic (no time-fix!)'),
        # Stage-A switches: '' = use the platform YAML value
        DeclareLaunchArgument('enable_filtering', default_value=''),
        DeclareLaunchArgument('enable_denoise', default_value=''),
        DeclareLaunchArgument('enable_rigidity', default_value=''),
        DeclareLaunchArgument('enable_dynamic', default_value=''),
        DeclareLaunchArgument('dynamic_debug', default_value=''),
        DeclareLaunchArgument('enable_landmarks', default_value='true'),
        DeclareLaunchArgument('enable_loop', default_value='true'),
        DeclareLaunchArgument('loop_debug', default_value='false'),
        DeclareLaunchArgument('enable_router', default_value='true'),
        DeclareLaunchArgument('enable_backend', default_value='true',
                              description='false = pose graph off (backend still publishes the outputs)'),
        DeclareLaunchArgument('enable_heading', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('publish_map', default_value='',
                              description="false = no /lio/map, /lio/cloud_effected, body scan ('' = YAML)"),
    ]

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', join(pkg, 'config', 'copernicus_localization.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup),
                                     rviz_node])
