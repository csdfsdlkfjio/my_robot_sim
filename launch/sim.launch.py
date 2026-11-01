import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    pkg_path = get_package_share_directory('my_robot_sim')
    xacro_file = os.path.join(pkg_path, 'urdf', 'my_car.urdf')
    robot_description_raw = xacro.process_file(xacro_file).toxml()

    gazebo_share = get_package_share_directory('gazebo_ros')
    default_world = os.path.join(gazebo_share, 'worlds', 'empty.world')

    # 直接用 gzserver/gzclient：gazebo.launch.py 不会转发 world 参数，
    # 那样永远只会启动空世界，SLAM 拿不到任何回波
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            gazebo_share, 'launch', 'gzserver.launch.py')]),
        launch_arguments={'world': LaunchConfiguration('world')}.items(),
    )

    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            gazebo_share, 'launch', 'gzclient.launch.py')]),
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_raw,
            'use_sim_time': True,
        }],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        # 车体静止时 base_link 离地 0.2m，从 0.25m 放下只掉 5cm，
        # 太高（比如 0.5m）会摔歪，万向轮着不了地就再也站不平
        arguments=[
            '-topic', 'robot_description', '-entity', 'my_car', '-z', '0.25'],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=default_world,
            description='Gazebo world file; use demo.world for SLAM'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Set to false to run gzserver only'),
        gzserver,
        gzclient,
        node_robot_state_publisher,
        spawn_entity,
    ])
