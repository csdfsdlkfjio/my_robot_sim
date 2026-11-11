"""Unit tests for the mobile-manipulation state machine helpers."""

import math
import os
import types

import pytest
from geometry_msgs.msg import Quaternion, Twist
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from moveit_msgs.msg import MoveItErrorCodes
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray

from my_robot_sim import mobile_manipulation_node as manipulation_module
from my_robot_sim.grasp_box import load_config
from my_robot_sim.mobile_manipulation_node import (
    ARM_BASE_FRAME,
    ARM_GROUP,
    ARM_JOINTS,
    ARM_MOUNT_OFFSET,
    ARM_TRAJECTORY_ACTION,
    BOX_OBJECT_ID,
    DRIVE_DISTANCE,
    END_EFFECTOR_LINK,
    FORWARD_SPEED,
    GAZEBO_HAND_LINK,
    GRIPPER_DOWN,
    GRIPPER_GOAL_TOLERANCE,
    GRIPPER_OPEN,
    GROUND_TRUTH_TOPIC,
    HAND_EFFORT_CONTROLLER,
    HAND_GROUP,
    HAND_JOINTS,
    HAND_POSITION_CONTROLLER,
    HOME_DURATION,
    HOME_JOINTS,
    MobileManipulationNode,
    ODOM_FRAME,
    PRE_CLOSE_CLEARANCE,
    PRE_CLOSE_MARGIN,
    PRE_CLOSE_MOVE_DURATION,
    SQUEEZE_CONTACT_TOLERANCE,
    SQUEEZE_SEAT_PENETRATION,
    TOUCH_LINKS,
    cube_diagonal,
    default_grasp_config_path,
    finger_gap,
    gripper_stuck_open,
    has_all_joints,
    inverse_quaternion,
    make_attach_command,
    make_attached_object,
    make_box_object,
    make_cartesian_request,
    make_effort_command,
    make_forward_twist,
    make_gripper_goal,
    make_gripper_trajectory,
    make_home_trajectory,
    make_open_effort_command,
    make_pose,
    make_reach_goal,
    make_release_command,
    make_removed_object,
    make_stop_twist,
    make_switch_request,
    move_should_retry,
    odom_distance,
    odom_xy,
    pads_hold_a_cube,
    point_in_arm,
    point_in_world,
    pre_close_position,
    pre_close_position_for,
    quaternion_multiply,
    rotate_vector,
    squeeze_outcome,
    squeeze_seat_window,
    tcp_hand_position,
    wait_for_pad_motion,
)


def test_arm_and_hand_joint_sets_are_complete_and_disjoint():
    """The controller split must cover 7 arm joints and 2 finger joints."""
    assert ARM_JOINTS == tuple('panda_joint%d' % i for i in range(1, 8))
    assert HAND_JOINTS == ('panda_finger_joint1', 'panda_finger_joint2')
    assert set(ARM_JOINTS) & set(HAND_JOINTS) == set()
    assert len(HOME_JOINTS) == 7


def test_drive_parameters_match_the_request():
    """The delivered flow drives 1 m and then picks up the cube."""
    assert DRIVE_DISTANCE == 1.0
    assert FORWARD_SPEED > 0.0
    assert GRIPPER_OPEN == 0.04


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


def test_tcp_hand_position_puts_the_hand_above_the_grasp_point():
    """With the gripper down the hand origin is tcp_offset above the box."""
    box = (0.49, 0.0, 0.25)
    assert tcp_hand_position(box, 0.10) == pytest.approx(
        (0.49, 0.0, 0.35))
    assert tcp_hand_position(box, 0.10, 0.075) == pytest.approx(
        (0.49, 0.0, 0.425))


def test_pre_close_position_leaves_the_pads_outside_the_cube():
    """The gap is twice the joint value; each pad clears a face by 12 mm."""
    assert pre_close_position(0.05) == pytest.approx(0.037)
    assert 2.0 * pre_close_position(0.05) - 0.05 == pytest.approx(
        2.0 * PRE_CLOSE_CLEARANCE)
    assert pre_close_position(0.05) < 0.04


def test_pre_close_position_for_shortens_the_press_travel():
    """With the cube measured 2 mm off centre, the near pad starts 4 mm out."""
    value = pre_close_position_for(0.05, 0.002)
    assert value == pytest.approx(0.025 + 0.002 + PRE_CLOSE_MARGIN)
    # The near pad's distance to the near face is exactly the margin.
    assert value - (0.025 + 0.002) == pytest.approx(PRE_CLOSE_MARGIN)
    # The far pad is 2 * offset + margin away, still outside the cube.
    assert value - (0.025 - 0.002) > PRE_CLOSE_MARGIN


def test_pre_close_position_for_never_exceeds_the_open_limit():
    """A large offset must not command a finger past its 0.04 m stop."""
    assert pre_close_position_for(0.05, 0.0) == pytest.approx(
        0.025 + PRE_CLOSE_MARGIN)
    assert pre_close_position_for(0.05, -0.02) == pytest.approx(
        0.025 + PRE_CLOSE_MARGIN)
    assert pre_close_position_for(0.05, 0.02) == pytest.approx(0.04)


def test_cube_diagonal_is_the_widest_span_of_the_cube():
    """The measured 7.19 cm failure gap was the diagonal of a 5 cm cube."""
    assert cube_diagonal(0.05) == pytest.approx(0.0707107, abs=1e-6)
    assert cube_diagonal(0.05) > 0.05


def test_squeeze_outcome_recognizes_the_face_grip():
    """Pads on opposite faces leave the gap equal to the cube width."""
    assert squeeze_outcome((0.025, 0.025), 0.05) == 'held'
    assert squeeze_outcome((0.0331, 0.0151), 0.05) == 'held'


def test_squeeze_outcome_recognizes_the_corner_grip():
    """A cube turned 45 degrees is pinched across its diagonal."""
    half_diagonal = cube_diagonal(0.05) / 2.0
    assert squeeze_outcome((half_diagonal, half_diagonal), 0.05) == 'corners'
    # The reported failure: (0.0358, 0.0361), gap 0.0719 m.
    assert squeeze_outcome((0.0358, 0.0361), 0.05) == 'corners'


def test_squeeze_outcome_recognizes_a_missed_cube():
    """Without the cube between them, both fingers close to zero."""
    assert squeeze_outcome((0.0, 0.0), 0.05) == 'missed'
    assert squeeze_outcome((0.002, 0.001), 0.05) == 'missed'


def test_squeeze_outcome_reports_an_unknown_blockage():
    """A gap that is neither the width nor the diagonal is still useful."""
    assert squeeze_outcome((0.018, 0.017), 0.05) == 'blocked'


def test_squeeze_outcome_accepts_the_measured_seat_depth():
    """
    Every live seat left the gap at 0.0440-0.0443 m for the 0.05 m cube.

    Those runs squeezed for the whole 10 N press, so the pads really were on
    the cube; the gap is short of the cube width because ODE lets the pads
    sink in. The old window started at 0.0440 m, so the margin was zero and a
    slightly deeper seat would have thrown the grasp away as 'blocked'.
    """
    assert squeeze_outcome((0.0221, 0.0220), 0.05) == 'held'
    assert squeeze_outcome((0.0219, 0.0218), 0.05) == 'held'


def test_squeeze_seat_window_only_widens_the_low_side():
    """Penetration may shorten the gap; nothing may widen it past tolerance."""
    low, high = squeeze_seat_window(0.05)
    assert low == pytest.approx(
        0.05 - 2.0 * (SQUEEZE_CONTACT_TOLERANCE + SQUEEZE_SEAT_PENETRATION))
    assert high == pytest.approx(
        0.05 + 2.0 * SQUEEZE_CONTACT_TOLERANCE)
    # Still far below the corner grip, so a rotated cube stays detectable.
    assert high < cube_diagonal(0.05) - 2.0 * SQUEEZE_CONTACT_TOLERANCE
    # A pad that reached the middle of the cube is not a face grip either.
    assert not pads_hold_a_cube((0.018, 0.018), 0.05)


def test_effort_command_presses_both_fingers_inward():
    """Both finger axes point outward, so a negative effort closes them."""
    message = make_effort_command(2.0)
    assert message.data == pytest.approx([-2.0, -2.0])
    # The sign is forced even if a positive magnitude is passed.
    assert make_effort_command(-2.0).data == pytest.approx([-2.0, -2.0])


def test_open_effort_command_pushes_both_fingers_outward():
    """A positive effort drives both fingers against their open stops."""
    message = make_open_effort_command(10.0)
    assert message.data == pytest.approx([10.0, 10.0])
    assert make_open_effort_command(-10.0).data == pytest.approx([10.0, 10.0])


def test_switch_request_strictly_swaps_the_hand_controllers():
    """The swap must be strict so the squeeze never runs on a stale owner."""
    request = make_switch_request(
        activate=['panda_hand_effort_controller'],
        deactivate=['panda_hand_controller'])
    assert list(request.activate_controllers) == [
        'panda_hand_effort_controller']
    assert list(request.deactivate_controllers) == ['panda_hand_controller']
    assert request.strictness == request.STRICT
    assert request.timeout.sec == 5


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
    goal = make_reach_goal((0.485, 0.0, 0.115))
    assert goal.request.group_name == ARM_GROUP == 'panda_arm'
    assert goal.planning_options.plan_only is False
    assert len(goal.request.goal_constraints) == 1

    constraints = goal.request.goal_constraints[0]
    position = constraints.position_constraints[0]
    assert position.header.frame_id == ARM_BASE_FRAME == 'panda_link0'
    assert position.link_name == END_EFFECTOR_LINK == 'panda_hand'
    target = position.constraint_region.primitive_poses[0].position
    assert (target.x, target.y, target.z) == (0.485, 0.0, 0.115)
    assert position.constraint_region.primitives[0].type == \
        SolidPrimitive.SPHERE

    orientation = constraints.orientation_constraints[0]
    assert orientation.header.frame_id == ARM_BASE_FRAME
    assert orientation.link_name == END_EFFECTOR_LINK
    assert isinstance(orientation.orientation, Quaternion)
    assert (orientation.orientation.x, orientation.orientation.y,
            orientation.orientation.z, orientation.orientation.w) == \
        GRIPPER_DOWN


def test_make_pose_uses_the_gripper_down_orientation():
    """The Cartesian descend and lift need a fully specified hand pose."""
    pose = make_pose((0.438, -0.007, 0.115))
    assert (pose.position.x, pose.position.y, pose.position.z) == \
        (0.438, -0.007, 0.115)
    orientation = pose.orientation
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == \
        GRIPPER_DOWN


def test_quaternion_multiply_composes_yaw_with_gripper_down():
    """A base yaw must rotate the gripper target but keep it pointing down."""
    half = math.sqrt(0.5)
    yaw_90 = (0.0, 0.0, half, half)
    identity = (0.0, 0.0, 0.0, 1.0)
    assert quaternion_multiply(identity, GRIPPER_DOWN) == \
        pytest.approx(GRIPPER_DOWN)
    combined = quaternion_multiply(yaw_90, GRIPPER_DOWN)
    assert combined == pytest.approx((half, half, 0.0, 0.0))
    # The hand z axis still points at the floor; the hand x axis now points
    # along world +y as well, i.e. the pads are parallel to an x/y-aligned box.
    x, y, z, w = combined
    for vector, expected in (
            ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        vx, vy, vz = vector
        # v' = q * v * q^-1, expanded without a tf dependency.
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        rotated = (
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        )
        assert rotated == pytest.approx(expected, abs=1e-9)


def test_rotate_vector_inverts_with_the_conjugate():
    """The ground-truth point transform relies on these two primitives."""
    half = math.sqrt(0.5)
    yaw_90 = (0.0, 0.0, half, half)
    assert rotate_vector(yaw_90, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1e-9)
    assert inverse_quaternion(yaw_90) == pytest.approx((0.0, 0.0, -half, half))
    vector = (0.3, -0.2, 0.7)
    assert rotate_vector(inverse_quaternion(yaw_90),
                         rotate_vector(yaw_90, vector)) == \
        pytest.approx(vector, abs=1e-9)


def make_ground_truth(base_xyz, yaw):
    """One Odometry message as libgazebo_ros_p3d publishes it."""
    message = Odometry()
    message.pose.pose.position.x = base_xyz[0]
    message.pose.pose.position.y = base_xyz[1]
    message.pose.pose.position.z = base_xyz[2]
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def test_ground_truth_base_pose_gives_the_arm_frame_geometry():
    """Slip-free geometry: panda_link0 = base_link - (0.04, 0, -0.035)."""
    import rclpy
    rclpy.init()
    node = MobileManipulationNode()
    try:
        yaw = math.pi / 2.0
        node.ground_truth = make_ground_truth((1.0, 0.0, 0.2), yaw)
        # The cube is 0.5 m along world +y, which is straight ahead of the
        # 90-degree-yawed base, 5 cm above the base origin.
        point = node.point_in_arm_frame((1.0, 0.5, 0.25))
        assert (point.x, point.y, point.z) == pytest.approx(
            (0.5 - ARM_MOUNT_OFFSET[0],
             0.0 - ARM_MOUNT_OFFSET[1],
             0.05 - ARM_MOUNT_OFFSET[2]))

        orientation = node.orientation_in_arm_frame()
        expected = quaternion_multiply(
            inverse_quaternion(
                (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))),
            GRIPPER_DOWN)
        assert orientation == pytest.approx(expected)
        # The corrected orientation still points the hand at the floor.
        assert rotate_vector(orientation, (0.0, 0.0, 1.0)) == pytest.approx(
            (0.0, 0.0, -1.0), abs=1e-9)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_world_and_arm_points_are_inverse_with_the_mount_offset():
    """A panda_link0 point and its world point must round-trip exactly."""
    base = make_ground_truth((1.0, 0.5, 0.2), math.pi / 2.0).pose.pose
    local = (0.5, 0.0, 0.05)
    world = point_in_world(local, base)
    # (0.5 - 0.04, 0, 0.05 + 0.035) rotated 90 degrees, then translated.
    assert world == pytest.approx((1.0, 0.96, 0.285), abs=1e-9)
    assert point_in_arm(world, base) == pytest.approx(local, abs=1e-9)


def test_world_measurement_survives_a_base_slide():
    """A world-fixed cube moves in panda_link0 when the chassis creeps."""
    cube_world = (1.45, 0.0, 0.275)
    before = make_ground_truth((1.0, 0.0, 0.2), 0.0).pose.pose
    after = make_ground_truth((0.9, 0.0, 0.2), 0.0).pose.pose
    assert point_in_arm(cube_world, before) == pytest.approx(
        (0.49, 0.0, 0.04))
    assert point_in_arm(cube_world, after) == pytest.approx(
        (0.59, 0.0, 0.04))


def test_only_planning_errors_are_retried():
    """A collision-free replan is worthwhile; a control failure is not."""
    assert move_should_retry(MoveItErrorCodes.PLANNING_FAILED)
    assert move_should_retry(MoveItErrorCodes.INVALID_MOTION_PLAN)
    assert not move_should_retry(MoveItErrorCodes.SUCCESS)
    assert not move_should_retry(MoveItErrorCodes.CONTROL_FAILED)


def test_reach_goal_uses_the_base_yaw_corrected_orientation():
    """The pad alignment correction must reach both MoveIt and the poses."""
    half = math.sqrt(0.5)
    corrected = quaternion_multiply((0.0, 0.0, half, half), GRIPPER_DOWN)
    goal = make_reach_goal((0.485, 0.0, 0.115), corrected)
    orientation = goal.request.goal_constraints[0].orientation_constraints[0]
    assert (orientation.orientation.x, orientation.orientation.y,
            orientation.orientation.z, orientation.orientation.w) == \
        pytest.approx(corrected)

    pose = make_pose((0.438, -0.007, 0.115), corrected)
    assert (pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w) == \
        pytest.approx(corrected)


def test_cartesian_request_plans_a_straight_line_for_the_hand():
    """A free-space RRT path can clip the cube; the descend must be straight."""
    waypoint = make_pose((0.438, -0.007, 0.115))
    request = make_cartesian_request([waypoint])
    assert request.header.frame_id == ARM_BASE_FRAME
    assert request.group_name == ARM_GROUP
    assert request.link_name == END_EFFECTOR_LINK
    assert request.avoid_collisions is True
    assert request.max_step > 0.0
    assert request.jump_threshold == 0.0
    assert len(request.waypoints) == 1
    assert request.waypoints[0].position.z == 0.115
    assert request.max_velocity_scaling_factor == pytest.approx(0.3)
    # An empty start state makes MoveIt start from the current state.
    assert list(request.start_state.joint_state.name) == []


def test_cartesian_request_can_slow_the_lift_down():
    """A slow lift keeps the pads from slipping off the squeezed cube."""
    waypoint = make_pose((0.438, -0.007, 0.215))
    request = make_cartesian_request([waypoint], velocity=0.1)
    assert request.max_velocity_scaling_factor == pytest.approx(0.1)


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
    """Both finger joints get the same position."""
    goal = make_gripper_goal(0.023)
    assert goal.request.group_name == HAND_GROUP == 'hand'
    constraints = goal.request.goal_constraints[0]
    assert [c.joint_name for c in constraints.joint_constraints] == \
        list(HAND_JOINTS)
    assert [c.position for c in constraints.joint_constraints] == \
        [0.023, 0.023]
    assert 0.0 <= 0.023 <= 0.04

    opened = make_gripper_goal(GRIPPER_OPEN)
    positions = [
        c.position
        for c in opened.request.goal_constraints[0].joint_constraints]
    assert positions == [GRIPPER_OPEN, GRIPPER_OPEN]


def test_gripper_trajectory_moves_both_fingers_with_a_duration():
    """
    The pre-close goes straight to the hand JTC with an explicit duration.

    Measured live at y=+0.6 m: a MoveIt hand goal for 0.0299 m starting from
    the 0.0394 m open stop (9.5 mm of travel) was accepted and declared
    "Goal reached, success!" after 0.27 s without moving a finger, because
    the JTC's 1 cm goal tolerance already covered the target. A trajectory
    with an explicit duration is executed point by point instead.
    """
    assert PRE_CLOSE_MOVE_DURATION > 0.0
    goal = make_gripper_trajectory(0.0299)
    assert list(goal.trajectory.joint_names) == list(HAND_JOINTS)
    assert len(goal.trajectory.points) == 1
    point = goal.trajectory.points[0]
    assert list(point.positions) == [0.0299, 0.0299]
    duration = (point.time_from_start.sec
                + point.time_from_start.nanosec * 1e-9)
    assert duration == pytest.approx(PRE_CLOSE_MOVE_DURATION)


def test_gripper_goal_is_planned_within_a_millimetre():
    """
    A loose goal tolerance eats the pre-close margin.

    Measured live: the pre-close asked for 0.031 m and the pads stopped at
    0.0350 m while MoveIt still reported success, because the goal region was
    +/- 5 mm wide. The pads then had to travel 13 mm under 10 N, which is an
    impact on a 50 g cube instead of a squeeze.
    """
    assert GRIPPER_GOAL_TOLERANCE <= 0.001
    goal = make_gripper_goal(0.031)
    for constraint in goal.request.goal_constraints[0].joint_constraints:
        assert constraint.tolerance_above == pytest.approx(
            GRIPPER_GOAL_TOLERANCE)
        assert constraint.tolerance_below == pytest.approx(
            GRIPPER_GOAL_TOLERANCE)


def test_gripper_stuck_open_separates_a_dead_command_from_the_cube():
    """
    (0.0400, 0.0400) after a closing command means the pads never moved.

    The reported failure sampled exactly the open stop for the whole 5 s
    squeeze; no cube between the pads can do that, because at full open the
    pads are 0.08 m apart and the cube is 0.05 m wide.
    """
    assert gripper_stuck_open((0.0400, 0.0400), 0.031)
    assert gripper_stuck_open((0.0399, 0.0398), 0.025)
    # The pads did move towards the target: the cube or something else is
    # holding them, which is a different diagnosis.
    assert not gripper_stuck_open((0.0350, 0.0350), 0.031)
    assert not gripper_stuck_open((0.0, 0.0), 0.025)
    # A goal within 2 mm of the stop is not judged: there is no travel to lose.
    assert not gripper_stuck_open((0.0399, 0.0399), 0.0395)


def test_wait_for_pad_motion_tolerates_a_late_controller():
    """
    Measured: the pre-close goal returned while both pads still read 0.0394.

    A single read after 0.2 s called a working controller dead. The helper
    must keep reading until the pads leave the open stop.
    """
    reads = [(0.0394, 0.0394)] * 8 + [(0.0350, 0.0350)]
    state = {'i': 0, 't': 0.0}

    def read():
        value = reads[min(state['i'], len(reads) - 1)]
        state['i'] += 1
        return value

    def sleep(seconds):
        state['t'] += seconds

    positions = wait_for_pad_motion(
        read, 0.035, timeout=3.0, poll=0.05,
        clock=lambda: state['t'], sleep=sleep)
    assert positions == pytest.approx((0.0350, 0.0350))
    # It waited for the motion instead of accepting the first read.
    assert state['i'] > 8


def test_wait_for_pad_motion_gives_up_on_a_dead_command_path():
    """A controller that never moves must still report the open stop."""
    reads = [(0.0399, 0.0398)]
    state = {'t': 0.0}

    positions = wait_for_pad_motion(
        lambda: reads[0], 0.031, timeout=1.0, poll=0.05,
        clock=lambda: state['t'],
        sleep=lambda seconds: state.__setitem__('t', state['t'] + seconds))
    assert positions == pytest.approx((0.0399, 0.0398))
    assert state['t'] == pytest.approx(1.0)


def test_wait_for_pad_motion_returns_immediately_when_the_pads_moved():
    """The common case must not add any wait to the grasp."""
    calls = {'n': 0}

    def read():
        calls['n'] += 1
        return (0.0360, 0.0360)

    positions = wait_for_pad_motion(
        read, 0.035, timeout=3.0, clock=lambda: 0.0,
        sleep=lambda seconds: pytest.fail('must not sleep'))
    assert positions == pytest.approx((0.0360, 0.0360))
    assert calls['n'] == 1


class FakeLogger:
    """Swallow the node's log calls."""

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class FakeServiceFuture:
    """Already-complete service future carrying a ListControllers response."""

    def __init__(self, states):
        self._response = types.SimpleNamespace(
            controller=[
                types.SimpleNamespace(name=name, state=state)
                for name, state in states.items()])

    def done(self):
        return True

    def result(self):
        return self._response


class FakeListClient:
    """controller_manager list_controllers stand-in."""

    def __init__(self, states):
        self.states = states

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        return FakeServiceFuture(self.states)


def make_controller_node(monkeypatch, states):
    """Node with just the state query and a recording controller switch."""
    node = object.__new__(MobileManipulationNode)
    node.list_client = FakeListClient(states)
    node.get_logger = lambda: FakeLogger()
    switches = []

    def switch(activate, deactivate, timeout=15.0):
        switches.append((list(activate), list(deactivate)))
        return True

    node.switch_controllers = switch
    monkeypatch.setattr(
        manipulation_module.rclpy, 'spin_until_future_complete',
        lambda *args, **kwargs: None)
    return node, switches


def test_pre_close_leaves_an_active_position_controller_alone(monkeypatch):
    """The normal launch state needs no switch."""
    node, switches = make_controller_node(monkeypatch, {
        HAND_POSITION_CONTROLLER: 'active',
        HAND_EFFORT_CONTROLLER: 'inactive',
    })

    assert node.ensure_hand_position_controller()
    assert switches == []


def test_pre_close_takes_the_fingers_back_from_the_effort_controller(
        monkeypatch):
    """
    A failed squeeze must not strand the pre-close on an inactive JTC.

    ensure_gripper_open() returns early when the pads already read open, so
    after a failed squeeze the effort controller can still own the finger
    interfaces when the next attempt pre-closes.
    """
    node, switches = make_controller_node(monkeypatch, {
        HAND_POSITION_CONTROLLER: 'inactive',
        HAND_EFFORT_CONTROLLER: 'active',
    })

    assert node.ensure_hand_position_controller()
    assert switches == [
        ([HAND_POSITION_CONTROLLER], [HAND_EFFORT_CONTROLLER])]


def test_pre_close_refuses_a_missing_position_controller(monkeypatch):
    """Without panda_hand_controller there is no way to close the pads."""
    node, switches = make_controller_node(monkeypatch, {
        HAND_EFFORT_CONTROLLER: 'active',
    })

    assert not node.ensure_hand_position_controller()
    assert switches == []


def test_pre_close_switches_the_position_controller_on_first():
    """The switch must happen before the trajectory, then squeeze re-swaps."""
    path = os.path.join(
        os.path.dirname(__file__), '..', 'my_robot_sim',
        'mobile_manipulation_node.py')
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    pre_close = source[
        source.index('    def pre_close_gripper'):
        source.index('    def press_until_seated')]
    assert 'self.ensure_hand_position_controller()' in pre_close
    assert pre_close.index('ensure_hand_position_controller') < \
        pre_close.index('send_hand_trajectory')
    squeeze = source[
        source.index('    def squeeze_gripper'):
        source.index('    def release_grip')]
    assert 'activate=[HAND_EFFORT_CONTROLLER]' in squeeze
    assert 'deactivate=[HAND_POSITION_CONTROLLER]' in squeeze


def test_both_grasp_sequences_verify_the_pre_close_position():
    """
    The pre-close must be read back instead of trusting the goal result.

    A goal can be planned, reported as reached, and still leave the pads at
    the open stop (measured), so the squeeze would press against a gripper
    that never closed. Both run() sequences go through the verifying helper.
    """
    for module in ('mobile_manipulation_node.py', 'visual_grasp.py'):
        path = os.path.join(
            os.path.dirname(__file__), '..', 'my_robot_sim', module)
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        assert 'if not self.pre_close_gripper(pre_close):' in source, module
        # The MoveIt hand goal can be swallowed by the JTC tolerance; the
        # pre-close must not fall back to it.
        assert 'self.send_move_goal(make_gripper_goal(pre_close))' \
            not in source, module
    path = os.path.join(
        os.path.dirname(__file__), '..', 'my_robot_sim',
        'mobile_manipulation_node.py')
    with open(path, encoding='utf-8') as handle:
        source = handle.read()
    assert 'self.send_hand_trajectory(position)' in source, path


def test_finger_gap_adds_both_pad_displacements():
    """The pads' inner faces are the sum of the two finger joints apart."""
    assert finger_gap((0.025, 0.025)) == pytest.approx(0.05)
    assert finger_gap((0.0331, 0.0151)) == pytest.approx(0.0482)


def test_symmetric_squeeze_holds_the_cube():
    """A cube centred between the pads leaves both fingers at size / 2."""
    assert pads_hold_a_cube((0.025, 0.025), 0.05)


def test_off_centre_cube_still_counts_as_gripped():
    """
    Measured in Gazebo: one finger contacts first and pushes the cube aside.

    (0.0331, 0.0151) is a real squeeze from the 4 N run; the gap still equals
    the cube width, so the pads straddle the cube.
    """
    assert pads_hold_a_cube((0.0331, 0.0151), 0.05)


def test_fully_closed_fingers_mean_the_cube_was_missed():
    """Without the cube between them, the fingers close all the way."""
    assert not pads_hold_a_cube((0.0, 0.0), 0.05)
    assert not pads_hold_a_cube((0.002, 0.001), 0.05)


def test_pads_that_never_reach_the_cube_do_not_count():
    """An open gripper (or a cube far wider than the pads) is not a grasp."""
    assert not pads_hold_a_cube((0.04, 0.04), 0.05)


def test_box_object_is_a_primitive_at_the_given_pose():
    """The cube must appear in RViz as a box at the measured box pose."""
    message = make_box_object((0.49, 0.0, 0.25), (0.05, 0.05, 0.05))
    assert isinstance(message, CollisionObject)
    assert message.id == BOX_OBJECT_ID == 'grasp_box'
    assert message.header.frame_id == ARM_BASE_FRAME
    assert message.operation == CollisionObject.ADD
    assert len(message.primitives) == 1
    assert message.primitives[0].type == SolidPrimitive.BOX
    assert list(message.primitives[0].dimensions) == [0.05, 0.05, 0.05]
    pose = message.primitive_poses[0]
    assert (pose.position.x, pose.position.y, pose.position.z) == \
        (0.49, 0.0, 0.25)
    assert pose.orientation.w == 1.0


def test_removed_object_only_carries_the_id_and_operation():
    """Removing the cube before closing the fingers needs no geometry."""
    message = make_removed_object(BOX_OBJECT_ID)
    assert message.id == BOX_OBJECT_ID
    assert message.operation == CollisionObject.REMOVE
    assert message.header.frame_id == ARM_BASE_FRAME


def test_attached_object_hangs_the_cube_on_the_hand():
    """The attached object is what makes MoveIt and RViz carry the cube."""
    box = make_box_object((0.49, 0.0, 0.25), (0.05, 0.05, 0.05))
    attached = make_attached_object(box)
    assert isinstance(attached, AttachedCollisionObject)
    assert attached.link_name == END_EFFECTOR_LINK == 'panda_hand'
    assert attached.object.id == BOX_OBJECT_ID
    assert set(attached.touch_links) == set(TOUCH_LINKS)
    assert 'panda_hand' in attached.touch_links
    assert 'panda_leftfinger' in attached.touch_links
    assert 'panda_rightfinger' in attached.touch_links


def test_attach_command_names_the_hand_and_the_gazebo_cube():
    """
    The Gazebo plugin welds grasp_box::link to the real hand link.

    Gazebo's URDF->SDF conversion lumps panda_hand into panda_link7, so a
    command naming panda_hand is rejected by the plugin.
    """
    assert GAZEBO_HAND_LINK == 'panda_link7'
    assert make_attach_command() == \
        'attach mobile_manipulator panda_link7 grasp_box link'


def test_attach_command_accepts_explicit_entities():
    """The command format is 'attach <parent> <link> <child> <link>'."""
    assert make_attach_command(
        'robot', 'tool', 'cube', 'body') == 'attach robot tool cube body'


def test_release_command_stops_pressing_both_fingers():
    """The weld, not the 10 N squeeze, carries the cube after the attach."""
    message = make_release_command()
    assert isinstance(message, Float64MultiArray)
    assert len(message.data) == len(HAND_JOINTS) == 2
    assert list(message.data) == [0.0, 0.0]


def test_both_grasp_sequences_release_the_squeeze_before_lifting():
    """
    Keeping 10 N on an already welded cube makes the finger chatter.

    Measured live: with the cube welded to the hand the effort press drives
    the pad into the rigid cube and the stiff contact bounces it back out, so
    the finger alternates between 0.0179 m and 0.0240 m at the control rate.
    The wrist camera shows that as the white pad jumping left and right at the
    cube's edge. Both run() sequences must stop the press after the weld and
    before the lift.
    """
    lift_markers = {
        'mobile_manipulation_node.py': '第九步：垂直向上抬起',
        'visual_grasp.py': 'lift = lifted_position(grasp',
    }
    for module, lift_marker in lift_markers.items():
        path = os.path.join(
            os.path.dirname(__file__), '..', 'my_robot_sim', module)
        with open(path, encoding='utf-8') as handle:
            source = handle.read()
        attach = source.index('if not self.attach_box_in_gazebo():')
        release = source.index('self.release_grip()')
        lift = source.index(lift_marker)
        assert attach < release < lift, module


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


def test_default_grasp_config_is_the_installed_file():
    path = default_grasp_config_path()
    assert os.path.isfile(path)
    assert path.endswith(os.path.join('config', 'grasp_box.yaml'))


def test_node_loads_the_grasp_config_and_exposes_parameters():
    """The node must read the same file the launch spawns the box from."""
    import rclpy
    rclpy.init()
    node = MobileManipulationNode()
    try:
        assert node.drive_distance == DRIVE_DISTANCE
        # The distance parameter must not shadow the driving method.
        assert callable(node.drive_forward)
        assert not callable(node.drive_distance)
        assert node.forward_speed == FORWARD_SPEED
        assert node.home_first is True
        assert node.grasp_config_path == default_grasp_config_path()
        assert node.grasp == load_config(default_grasp_config_path())
        assert node.grasp['size'] == 0.05
        assert ODOM_FRAME == 'odom'
        # move_group's scene monitor listens on these two topics; the
        # /apply_planning_scene service cannot carry attached objects.
        assert node.collision_object_pub.topic_name == '/collision_object'
        assert node.attached_pub.topic_name == '/attached_collision_object'
        assert node.hand_effort_pub.topic_name == \
            '/panda_hand_effort_controller/commands'
        assert node.switch_client.srv_name == \
            '/controller_manager/switch_controller'
        assert node.grasp['squeeze_effort'] > 0.0
        assert isinstance(make_stop_twist(), Twist)
        # Without the plugin the node must fall back to the odom TF.
        assert node.ground_truth is None
        assert any(
            subscription.topic_name == GROUND_TRUTH_TOPIC
            for subscription in node.subscriptions)
    finally:
        node.destroy_node()
        rclpy.shutdown()
