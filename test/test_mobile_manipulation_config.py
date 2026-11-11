"""Consistency checks for controllers, SRDF and the bringup launch file."""

import os
import xml.etree.ElementTree as ET

import pytest
import yaml

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONTROLLERS = os.path.join(
    PKG_ROOT, 'config', 'mobile_manip_controllers.yaml')
MOVEIT_CONTROLLERS = os.path.join(
    PKG_ROOT, 'config', 'mobile_manip_moveit_controllers.yaml')
SRDF = os.path.join(PKG_ROOT, 'config', 'mobile_manip.srdf')
LAUNCH = os.path.join(
    PKG_ROOT, 'launch', 'mobile_manipulation.launch.py')

WHEEL_JOINTS = {'left_wheel_joint', 'right_wheel_joint'}


def load_yaml(path):
    """Read a YAML file."""
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_ros2_control_controller_joints():
    """Arm and hand controllers must cover exactly the Panda joints."""
    config = load_yaml(CONTROLLERS)
    manager = config['controller_manager']['ros__parameters']
    assert manager['panda_arm_controller']['type'] == \
        'joint_trajectory_controller/JointTrajectoryController'
    assert manager['panda_hand_controller']['type'] == \
        'joint_trajectory_controller/JointTrajectoryController'
    assert manager['joint_state_broadcaster']['type'] == \
        'joint_state_broadcaster/JointStateBroadcaster'

    arm = set(config['panda_arm_controller']['ros__parameters']['joints'])
    hand = set(config['panda_hand_controller']['ros__parameters']['joints'])
    assert arm == {'panda_joint%d' % index for index in range(1, 8)}
    assert hand == {'panda_finger_joint1', 'panda_finger_joint2'}
    assert arm & hand == set()
    assert (arm | hand) & WHEEL_JOINTS == set()


def test_moveit_controller_manager_matches_ros2_control():
    """The MoveIt manager must send trajectories to the two controllers."""
    config = load_yaml(MOVEIT_CONTROLLERS)
    manager = config['moveit_simple_controller_manager']
    assert set(manager['controller_names']) == {
        'panda_arm_controller', 'panda_hand_controller'}
    for name in manager['controller_names']:
        assert manager[name]['action_ns'] == 'follow_joint_trajectory'
        assert manager[name]['type'] == 'FollowJointTrajectory'

    ros2_control = load_yaml(CONTROLLERS)
    assert set(manager['panda_arm_controller']['joints']) == set(
        ros2_control['panda_arm_controller']['ros__parameters']['joints'])
    assert set(manager['panda_hand_controller']['joints']) == set(
        ros2_control['panda_hand_controller']['ros__parameters']['joints'])


def srdf_group(name):
    """Return an SRDF group element by name."""
    root = ET.parse(SRDF).getroot()
    return next(group for group in root.findall('group')
                if group.get('name') == name)


def test_srdf_arm_group_is_the_panda_chain():
    """The planning group must remain panda_link0 -> panda_link8."""
    arm = srdf_group('panda_arm')
    chain = arm.find('chain')
    assert chain.get('base_link') == 'panda_link0'
    assert chain.get('tip_link') == 'panda_link8'


def test_srdf_hand_group_matches_controller():
    """Both finger joints are active, matching panda_hand_controller."""
    hand = srdf_group('hand')
    active = {joint.get('name') for joint in hand.findall('joint')}
    passive = {joint.get('name') for joint in hand.findall('passive_joint')}
    controller = set(load_yaml(CONTROLLERS)[
        'panda_hand_controller']['ros__parameters']['joints'])
    assert active == controller
    assert active == {'panda_finger_joint1', 'panda_finger_joint2'}
    assert passive == set()


def test_hand_uses_position_and_effort_controllers():
    """Opening is position-controlled; the squeeze needs a real force."""
    config = load_yaml(CONTROLLERS)
    hand = config['panda_hand_controller']['ros__parameters']
    assert hand['command_interfaces'] == ['position']
    assert set(hand['state_interfaces']) == {'position', 'velocity'}

    manager = config['controller_manager']['ros__parameters']
    assert manager['panda_hand_effort_controller']['type'] == \
        'effort_controllers/JointGroupEffortController'
    effort = config['panda_hand_effort_controller']['ros__parameters']
    assert set(effort['joints']) == {
        'panda_finger_joint1', 'panda_finger_joint2'}


def test_hand_position_goal_tolerance_is_tighter_than_the_pre_close_travel():
    """
    The JTC must not mistake the open stop for the pre-close target.

    Live at y=+0.6 m the pads sat at 0.0392/0.0395 m and the target was
    0.0295 m; with a 1 cm per-joint goal tolerance the controller reported
    success immediately and never moved a finger, three resends included.
    The squeeze runs on the effort controller, so the position JTC no longer
    needs that tolerance. Keep it well below the smallest pre-close travel
    (5 cm cube, no lateral offset: 0.040 - 0.029 = 11 mm).
    """
    config = load_yaml(CONTROLLERS)
    constraints = config['panda_hand_controller']['ros__parameters'][
        'constraints']
    pre_close_travel = 0.040 - (0.05 / 2.0 + 0.004)
    assert pre_close_travel == pytest.approx(0.011)
    for joint in ('panda_finger_joint1', 'panda_finger_joint2'):
        goal = constraints[joint]['goal']
        assert 0.0 < goal <= 0.002, joint
        assert goal <= pre_close_travel / 4.0, joint


def test_srdf_robot_name_matches_the_merged_urdf():
    """The SRDF and URDF robot names must match or MoveIt logs an error."""
    urdf = ET.parse(os.path.join(PKG_ROOT, 'urdf',
                                 'mobile_manipulator.urdf.xacro')).getroot()
    srdf = ET.parse(SRDF).getroot()
    assert srdf.get('name') == urdf.get('name') == 'mobile_manipulator'


def test_srdf_has_no_virtual_joint_to_panda_link0():
    """The merged URDF already fixes panda_link0 to the base."""
    root = ET.parse(SRDF).getroot()
    assert root.find('virtual_joint') is None


def test_launch_starts_gazebo_ros2_control_and_moveit():
    """The one-shot launch must reference the model and all subsystems."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    for token in (
        'mobile_manipulator.urdf.xacro',
        'mobile_manip.srdf',
        'mobile_manip_moveit_controllers.yaml',
        'gzserver.launch.py',
        'spawn_entity.py',
        'controller_manager',
        'moveit_ros_move_group',
        'MoveItConfigsBuilder',
        'panda_arm_controller',
        'panda_hand_controller',
        "name='mobile_robot_state_publisher'",
        'grasp_box.yaml',
        'grasp_box',
        'box_sdf',
    ):
        assert token in source, 'launch is missing %s' % token
    # The table is a static model in demo.world, not a launch spawn: SLAM
    # maps the world without the launch, and a spawned table would be
    # missing from the map it is meant to be avoided in.
    assert 'grasp_table' not in source
    assert 'pedestal_sdf' not in source
    # The ros2_control YAML is referenced from the model's Gazebo plugin,
    # which test_mobile_manipulation_model already checks.


def test_launch_spawns_the_cube_on_the_world_table():
    """The table comes from demo.world; only the cube is spawned."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    start = source.index('box_spawner = Node(')
    block = source[start:source.index('TimerAction(', start)]
    for token in ("'-entity', 'grasp_box'", "'-file', grasp_file"):
        assert token in block
    # The initial z must be the table top plus half the cube, otherwise the
    # cube falls a few centimetres or spawns inside the table.
    assert "grasp['z']" in block


def test_launch_waits_for_gazebo_before_spawning():
    """spawn_entity.py gives up after 30 s; a cold gzserver can be slower."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    start = source.index('spawn_entities = TimerAction(')
    block = source[start:source.index(')', start)]
    assert 'period=5.0' in block
    # Both the robot and the cube must wait for gzserver.
    assert 'spawn_entity' in block
    assert 'box_spawner' in block
    # The timer itself has to be part of the launched description.
    assert '        spawn_entities,' in source


def test_launch_loads_the_effort_controller_inactive():
    """switch_controller can only activate a controller that is loaded."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    assert "'panda_hand_effort_controller'" in source
    assert "'--inactive'" in source


def test_launch_sanitizes_the_robot_description():
    """A raw description breaks gazebo_ros2_control's --param rule."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    # Comments must go: a ': ' or a '*'-leading line inside one makes rcl
    # reject the --param override, and the plugin then starts without a
    # controller manager.
    assert "re.sub(r'<!--.*?-->', '', description" in source
    assert "' '.join(" in source
    # move_group's internal robot_state_publisher must serve the same text,
    # otherwise which node answers the plugin's parameter request decides
    # whether the controller manager starts.
    assert \
        "moveit_config.robot_description['robot_description'] = " in source


def test_launch_does_not_start_the_task_node():
    """The task is run in its own terminal, matching the documented flow."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    assert "executable='mobile_manipulation_node'" not in source
