"""Unit tests for the mobile-manipulation state machine helpers."""

import math

from geometry_msgs.msg import Quaternion, Twist
from moveit_msgs.msg import MoveItErrorCodes
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from my_robot_sim.mobile_manipulation_node import (
    ARM_BASE_FRAME, ARM_GROUP, ARM_JOINTS, ARM_TRAJECTORY_ACTION,
    DRIVE_DISTANCE, END_EFFECTOR_LINK, FORWARD_SPEED, GRIPPER_CLOSED,
    GRIPPER_DOWN, GRIPPER_OPEN, HAND_GROUP, HAND_JOINTS, HOME_DURATION,
    HOME_JOINTS, MobileManipulationNode, TARGET_POSITION,
    has_all_joints, make_forward_twist, make_gripper_goal,
    make_home_trajectory, make_reach_goal, make_stop_twist,
    odom_distance, odom_xy,
)


def test_arm_and_hand_joint_sets_are_complete_and_disjoint():
    """The controller split must cover 7 arm joints and 2 finger joints."""
    assert ARM_JOINTS == tuple('panda_joint%d' % i for i in range(1, 8))
    assert HAND_JOINTS == ('panda_finger_joint1', 'panda_finger_joint2')
    assert set(ARM_JOINTS) & set(HAND_JOINTS) == set()
    assert len(HOME_JOINTS) == 7


def test_target_and_drive_distance_match_the_request():
    """The delivered flow is: drive 1 m, reach (0.5, 0, 0.3), close."""
    assert DRIVE_DISTANCE == 1.0
    assert TARGET_POSITION == (0.5, 0.0, 0.3)
    assert FORWARD_SPEED > 0.0


def test_gripper_orientation_is_unit_quaternion_pointing_down():
    """(x=1, w=0) rotates the hand z axis to -z: gripper looks down."""
    x, y, z, w = GRIPPER_DOWN
    assert abs(math.sqrt(x * x + y * y + z * z + w * w) - 1.0) < 1e-9
    local_z_in_base = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    assert local_z_in_base[2] == -1.0


def test_odom_distance_uses_planar_euclidean_distance():
    """A 3-4-5 displacement must report 5 metres."""
    assert odom_distance((1.0, 2.0), (4.0, 6.0)) == 5.0
    assert odom_distance((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_odom_xy_extracts_position():
    """odom_xy must read the pose position, not the twist."""
    message = Odometry()
    message.pose.pose.position.x = 1.25
    message.pose.pose.position.y = -0.5
    message.twist.twist.linear.x = 99.0
    assert odom_xy(message) == (1.25, -0.5)


def test_make_forward_and_stop_twists():
    """Forward command keeps angular.z at zero; stop command is all zeros."""
    forward = make_forward_twist(0.3)
    assert forward.linear.x == 0.3
    assert forward.angular.z == 0.0

    stop = make_stop_twist()
    assert stop.linear.x == 0.0
    assert stop.linear.y == 0.0
    assert stop.linear.z == 0.0
    assert stop.angular.x == 0.0
    assert stop.angular.y == 0.0
    assert stop.angular.z == 0.0


def test_reach_goal_targets_hand_pose_in_arm_base_frame():
    """The goal must be a position + orientation constraint for panda_hand."""
    goal = make_reach_goal()
    assert goal.request.group_name == ARM_GROUP == 'panda_arm'
    assert goal.planning_options.plan_only is False
    assert len(goal.request.goal_constraints) == 1

    constraints = goal.request.goal_constraints[0]
    position = constraints.position_constraints[0]
    assert position.header.frame_id == ARM_BASE_FRAME == 'panda_link0'
    assert position.link_name == END_EFFECTOR_LINK == 'panda_hand'
    target = position.constraint_region.primitive_poses[0].position
    assert (target.x, target.y, target.z) == TARGET_POSITION
    assert position.constraint_region.primitives[0].type == \
        SolidPrimitive.SPHERE

    orientation = constraints.orientation_constraints[0]
    assert orientation.header.frame_id == ARM_BASE_FRAME
    assert orientation.link_name == END_EFFECTOR_LINK
    assert isinstance(orientation.orientation, Quaternion)
    assert (orientation.orientation.x, orientation.orientation.y,
            orientation.orientation.z, orientation.orientation.w) == \
        GRIPPER_DOWN


def test_home_trajectory_commands_all_seven_joints():
    """Homing goes straight to the JTC so an invalid spawn state is fine."""
    goal = make_home_trajectory()
    assert ARM_TRAJECTORY_ACTION == \
        '/panda_arm_controller/follow_joint_trajectory'
    assert goal.trajectory.joint_names == list(ARM_JOINTS)
    assert len(goal.trajectory.points) == 1
    point = goal.trajectory.points[0]
    assert list(point.positions) == list(HOME_JOINTS)
    assert point.time_from_start.sec == int(HOME_DURATION)
    assert point.time_from_start.sec >= 3


def test_gripper_goal_commands_both_fingers_symmetrically():
    """Both finger joints get the same closing position."""
    goal = make_gripper_goal()
    assert goal.request.group_name == HAND_GROUP == 'hand'
    constraints = goal.request.goal_constraints[0]
    assert [c.joint_name for c in constraints.joint_constraints] == \
        list(HAND_JOINTS)
    assert [c.position for c in constraints.joint_constraints] == \
        [GRIPPER_CLOSED, GRIPPER_CLOSED]
    assert 0.0 <= GRIPPER_CLOSED <= 0.04

    opened = make_gripper_goal(GRIPPER_OPEN)
    positions = [
        c.position
        for c in opened.request.goal_constraints[0].joint_constraints]
    assert positions == [GRIPPER_OPEN, GRIPPER_OPEN]
    assert GRIPPER_OPEN == 0.04


def test_has_all_joints_requires_every_arm_joint():
    """The state machine must wait until MoveIt has all arm joint states."""
    state = JointState()
    state.name = list(ARM_JOINTS[:3])
    assert has_all_joints(state, ARM_JOINTS) is False
    state.name = list(ARM_JOINTS)
    assert has_all_joints(state, ARM_JOINTS) is True
    assert has_all_joints(None, ARM_JOINTS) is False


def test_success_code_is_moveit_success():
    """Guard against accidentally accepting a non-zero MoveIt error code."""
    assert MoveItErrorCodes.SUCCESS == 1


def test_node_declares_tunable_parameters():
    """The node exposes distance / speed / target overrides."""
    import rclpy
    rclpy.init()
    node = MobileManipulationNode()
    try:
        assert node.drive_distance == DRIVE_DISTANCE
        # The distance parameter must not shadow the driving method.
        assert callable(node.drive_forward)
        assert not callable(node.drive_distance)
        assert node.forward_speed == FORWARD_SPEED
        assert node.target_position == TARGET_POSITION
        assert node.home_first is True
        assert isinstance(make_stop_twist(), Twist)
    finally:
        node.destroy_node()
        rclpy.shutdown()
