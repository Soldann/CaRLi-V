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

        # # Launch the radar_cube_node
        launch_ros.actions.Node(
            package='carli_v',
            executable='radar_cube_node',
            name='radar_cube_node',
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
