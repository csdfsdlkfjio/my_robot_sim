import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('my_robot_sim')
    default_map = os.path.join(pkg_share, 'maps', 'my_map.yaml')
    nav2_share = get_package_share_directory('nav2_bringup')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            nav2_share, 'launch', 'bringup_launch.py')]),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': 'true',
            'autostart': 'true',
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(
            nav2_share, 'rviz', 'nav2_default_view.rviz')],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        # 地图默认用安装到 share 的那一份，可以这样覆盖：
        # ros2 launch my_robot_sim nav2.launch.py map:=/绝对路径/xxx.yaml
        DeclareLaunchArgument(
            'map', default_value=default_map,
            description='Occupancy grid yaml file used by map_server'),
        nav2_bringup,
        rviz,
    ])
