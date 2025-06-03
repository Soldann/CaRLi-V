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

        # Launch the radar_cube_node
        launch_ros.actions.Node(
            package='carli_v',
            executable='scene_flow_node',
            name='scene_flow_node',
            output='screen'
        ),
    ])
