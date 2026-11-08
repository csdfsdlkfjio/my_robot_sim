"""Consistency checks for controllers, SRDF and the bringup launch file."""

import os
import xml.etree.ElementTree as ET

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
    ):
        assert token in source, 'launch is missing %s' % token
    # The ros2_control YAML is referenced from the model's Gazebo plugin,
    # which test_mobile_manipulation_model already checks.


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
