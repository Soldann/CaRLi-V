import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        # Include the static_tf.launch.py file
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource([
                launch.substitutions.PathJoinSubstitution([
                    launch_ros.substitutions.FindPackageShare('carli_v'),
                    'launch',
                    'static_tf.launch.py'
                ])
            ])
        ),

        # Ablation study parameters for radar_cube_node
        launch.actions.DeclareLaunchArgument('intensity_threshold_offset', default_value='5.0'),
        launch.actions.DeclareLaunchArgument('morph_filter_elev', default_value='10'),
        launch.actions.DeclareLaunchArgument('morph_filter_azimuth', default_value='10'),
        launch.actions.DeclareLaunchArgument('morph_filter_range', default_value='20'),
        launch.actions.DeclareLaunchArgument('target_azimuth_bins', default_value='50'),
        launch.actions.DeclareLaunchArgument('target_elevation_bins', default_value='2'),

        # Launch the radar_cube_node
        launch_ros.actions.Node(
            package='carli_v',
            executable='radar_cube_node',
            name='radar_cube_node',
            parameters=[{
                'intensity_threshold_offset': launch.substitutions.LaunchConfiguration('intensity_threshold_offset'),
                'morph_filter_elev': launch.substitutions.LaunchConfiguration('morph_filter_elev'),
                'morph_filter_azimuth': launch.substitutions.LaunchConfiguration('morph_filter_azimuth'),
                'morph_filter_range': launch.substitutions.LaunchConfiguration('morph_filter_range'),
                'target_azimuth_bins': launch.substitutions.LaunchConfiguration('target_azimuth_bins'),
                'target_elevation_bins': launch.substitutions.LaunchConfiguration('target_elevation_bins'),
            }],
            output='screen'
        ),
        # Launch the optical_flow_node
        launch_ros.actions.Node(
            package='carli_v',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen'
        ),

        # Get parameters for radar_full_velocity_node
        launch.actions.DeclareLaunchArgument(
            'save_pcd_as',
            default_value='',
            description='If set, save the radar point clouds to a dataset with this name'
        ),

        # Launch the radar_full_velocity_node
        launch_ros.actions.Node(
            package='carli_v',
            executable='radar_full_velocity_node',
            name='radar_full_velocity_node',
            parameters=[{'save_pcd_as': launch.substitutions.LaunchConfiguration('save_pcd_as')}],
            output='screen'
        ),
    ])
