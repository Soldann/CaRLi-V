import launch
import launch_ros

def generate_launch_description():
    camera_name_arg = launch.actions.DeclareLaunchArgument(
        'camera_name',
        default_value='zed',
        description='Zed2i Camera Name'
    )
    zed2i_to_hesai_static_tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='zed2i_to_hesai_static_tf_publisher',
        arguments=['0.0069', '0.011', '0.11',
                   '1.5734143', '-0.010472', '0.3455752',
                   launch.substitutions.PythonExpression(['"', launch.substitutions.LaunchConfiguration("camera_name"), "_camera_link", '"']),
                   'hesai_lidar']
    )
    zed2i_to_vmd3_static_tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='zed2i_to_vmd3_static_tf_publisher',
        arguments=['0.03', '0.0', '-0.06',
                   '0.0', '0.0', '0.0',
                   launch.substitutions.PythonExpression(['"', launch.substitutions.LaunchConfiguration("camera_name"), "_camera_link", '"']),
                   'vmd3_radar']
    )
    return launch.LaunchDescription([
        camera_name_arg,
        zed2i_to_hesai_static_tf,
        zed2i_to_vmd3_static_tf
    ])