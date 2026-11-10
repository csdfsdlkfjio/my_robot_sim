"""
One-shot bringup: Gazebo + base + Panda ros2_control + MoveIt 2.

Usage:
    ros2 launch my_robot_sim mobile_manipulation.launch.py
    ros2 launch my_robot_sim mobile_manipulation.launch.py \
        gui:=false rviz:=false

The default world is worlds/demo.world: an enclosed 8 x 8 m room with four
walls, so navigation cannot drive the base off the ground plane.

The launch does NOT start the task node; run it separately:
    ros2 run my_robot_sim mobile_manipulation_node
"""

import os
import re
import tempfile

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

from my_robot_sim.grasp_box import (
    box_sdf,
    load_config,
    pedestal_pose,
    pedestal_sdf,
)


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
    # Default to the enclosed 8 x 8 m room. Gazebo's empty.world has no
    # walls and only a finite ground plane, so Nav2 can drive the base off
    # the edge and drop it into the void. Pass
    # world:=.../empty.world to get the old bare world back.
    default_world = os.path.join(pkg_share, 'worlds', 'demo.world')

    # gzserver resolves a URDF <plugin filename="libgazebo_grasp_attach.so"/>
    # on GAZEBO_PLUGIN_PATH. The grasp weld plugin lives in
    # my_robot_grasp_attach; without its lib directory gzserver silently skips
    # the plugin and the cube would never leave the pedestal.
    attach_lib_dir = os.path.join(
        get_package_prefix('my_robot_grasp_attach'), 'lib')
    if not os.path.isdir(attach_lib_dir):
        raise RuntimeError(
            'my_robot_grasp_attach is installed but %s does not exist; '
            'rebuild it with colcon build --packages-select '
            'my_robot_grasp_attach' % attach_lib_dir)
    gazebo_plugin_path = attach_lib_dir
    if os.environ.get('GAZEBO_PLUGIN_PATH'):
        gazebo_plugin_path += os.pathsep + os.environ['GAZEBO_PLUGIN_PATH']
    set_gazebo_plugin_path = SetEnvironmentVariable(
        'GAZEBO_PLUGIN_PATH', gazebo_plugin_path)

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

    # The node name is part of the contract: the URDF's gazebo_ros2_control
    # plugin reads robot_description from this node, and move_group already
    # runs a node called robot_state_publisher, so a unique name is needed.
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='mobile_robot_state_publisher',
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

    # The grasp cube and the pedestal holding it up are described by
    # config/grasp_box.yaml. spawn_entity.py only accepts files, so render the
    # SDF text once at launch time into a temporary directory.
    grasp = load_config(os.path.join(pkg_share, 'config', 'grasp_box.yaml'))
    grasp_sdf_dir = tempfile.mkdtemp(prefix='my_robot_sim_grasp_')
    grasp_files = {}
    for entity, builder in (('grasp_box', box_sdf),
                            ('grasp_pedestal', pedestal_sdf)):
        path = os.path.join(grasp_sdf_dir, entity + '.sdf')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(builder(grasp))
        grasp_files[entity] = path

    def grasp_spawner(entity, position):
        return Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', entity,
                '-file', grasp_files[entity],
                '-x', '%g' % position[0],
                '-y', '%g' % position[1],
                '-z', '%g' % position[2],
            ],
            output='screen',
        )

    # Spawners started together run in parallel. The cube is free to fall, so
    # if it reaches the world before the static pedestal it drops through the
    # empty space where the pedestal will appear and ends up embedded. Chain
    # the cube spawn on the pedestal spawner's exit so the support is there
    # first.
    pedestal_spawner = grasp_spawner(
        'grasp_pedestal', pedestal_pose(grasp))
    box_spawner = grasp_spawner(
        'grasp_box', (grasp['x'], grasp['y'], grasp['z']))
    spawn_box_after_pedestal = RegisterEventHandler(
        OnProcessExit(
            target_action=pedestal_spawner,
            on_exit=[box_spawner],
        ))

    # spawn_entity.py only waits 30 s for /spawn_entity to be advertised.
    # On a cold WSL2 start gzserver can need longer than that to load the
    # world (and to give up on the online model database), so every spawner
    # starts, times out with "Service /spawn_entity unavailable", and leaves
    # the robot and pedestal unspawned. Give Gazebo a five-second head start
    # before the spawners begin their own wait.
    spawn_entities = TimerAction(
        period=5.0,
        actions=[spawn_entity, pedestal_spawner],
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
        # The task node swaps the hand to this controller to press the cube
        # with a constant force. It has to be loaded (configured, inactive)
        # up front: switch_controller can only activate a loaded controller.
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'panda_hand_effort_controller',
                '-c', '/controller_manager',
                '--controller-manager-timeout', '60',
                '--inactive',
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
        set_gazebo_plugin_path,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_entities,
        spawn_box_after_pedestal,
        TimerAction(period=3.0, actions=controller_spawners),
        move_group,
        rviz,
    ])
