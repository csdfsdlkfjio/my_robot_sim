"""
One-shot bringup: Gazebo + base + Panda ros2_control + MoveIt 2.

Usage:
    ros2 launch my_robot_sim mobile_manipulation.launch.py
    ros2 launch my_robot_sim mobile_manipulation.launch.py \
        gui:=false rviz:=false

The launch does NOT start the task node; run it separately:
    ros2 run my_robot_sim mobile_manipulation_node
"""

import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_share = get_package_share_directory('my_robot_sim')
    xacro_file = os.path.join(
        pkg_share, 'urdf', 'mobile_manipulator.urdf.xacro')
    srdf_file = os.path.join(pkg_share, 'config', 'mobile_manip.srdf')
    moveit_controllers_file = os.path.join(
        pkg_share, 'config', 'mobile_manip_moveit_controllers.yaml')

    # Load kinematics / joint limits / OMPL configuration from the stock Panda
    # MoveIt package, but replace the robot description and SRDF with the
    # merged mobile-manipulator model. Absolute file paths are supported by
    # MoveItConfigsBuilder.
    moveit_config = (
        MoveItConfigsBuilder(
            'moveit_resources_panda',
            package_name='moveit_resources_panda_moveit_config',
        )
        .robot_description(file_path=xacro_file)
        .robot_description_semantic(file_path=srdf_file)
        .trajectory_execution(file_path=moveit_controllers_file)
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    # gazebo_ros2_control copies robot_state_publisher's description into the
    # controller_manager node as a ``--param robot_description:=...`` command
    # line override. rcl parses that value as YAML, so a multi-line value, a
    # line starting with '*', or a ``: `` inside an XML comment makes the
    # parse fail and leaves the plugin without a controller manager.
    #
    # move_group also runs an internal node called robot_state_publisher, so
    # the plugin's parameter request can be answered by either node. Hand both
    # the same comment-free single-line text so the answer never depends on
    # the race.
    description = moveit_config.robot_description['robot_description']
    robot_description = ' '.join(
        re.sub(r'<!--.*?-->', '', description, flags=re.S).split())
    moveit_config.robot_description['robot_description'] = robot_description

    gazebo_share = get_package_share_directory('gazebo_ros')
    default_world = os.path.join(gazebo_share, 'worlds', 'empty.world')

    # Same gzserver/gzclient split as sim.launch.py so the custom world is
    # actually forwarded (gazebo.launch.py drops the world argument).
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            gazebo_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': LaunchConfiguration('verbose'),
        }.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            gazebo_share, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(LaunchConfiguration('gui')),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'mobile_manipulator',
            '-z', '0.25',
        ],
        output='screen',
    )

    # gazebo_ros2_control creates /controller_manager inside gzserver. The
    # spawners wait for that service, so the timer only avoids noisy retries.
    controller_spawners = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'joint_state_broadcaster',
                '-c', '/controller_manager',
                '--controller-manager-timeout', '60',
            ],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'panda_arm_controller',
                '-c', '/controller_manager',
                '--controller-manager-timeout', '60',
            ],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'panda_hand_controller',
                '-c', '/controller_manager',
                '--controller-manager-timeout', '60',
            ],
            output='screen',
        ),
    ]

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), {'use_sim_time': True}],
    )

    rviz_config = os.path.join(
        get_package_share_directory('moveit_resources_panda_moveit_config'),
        'launch', 'moveit.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[moveit_config.to_dict(), {'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value=default_world,
            description='Gazebo world file'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Start gzclient (set false for headless runs)'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Start RViz with the MoveIt planning display'),
        DeclareLaunchArgument(
            'verbose', default_value='false',
            description='Run gzserver with verbose output'),
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_entity,
        TimerAction(period=3.0, actions=controller_spawners),
        move_group,
        rviz,
    ])
