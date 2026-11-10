"""Unit tests for the search-and-grasp orchestration helpers."""

import math
import os

import numpy as np
import pytest
from nav_msgs.msg import Odometry

from my_robot_sim.search_and_grasp import (
    GRASP_ATTEMPTS,
    MAP_FRAME,
    RETREAT_DISTANCE,
    SEARCH_CAMERA_TILT,
    SEARCH_STEP_ANGLE,
    SEARCH_STEPS,
    STANDOFF_DISTANCE,
    SearchAndGraspNode,
    make_navigate_goal,
    make_turn_twist,
    odom_yaw,
    quaternion_yaw,
    search_camera_quaternion,
    standoff_goal,
    wrap_angle,
    yaw_quaternion,
)
from my_robot_sim.visual_grasp import VisualGraspNode, quaternion_matrix

PKG_ROOT = os.path.join(os.path.dirname(__file__), '..')
MODULE = os.path.join(PKG_ROOT, 'my_robot_sim', 'search_and_grasp.py')


def test_wrap_angle_keeps_a_difference_in_one_turn():
    """Per-step yaw differences must not wrap around to +/- 2 pi."""
    assert wrap_angle(0.3) == pytest.approx(0.3)
    assert wrap_angle(3.0 * math.pi / 2.0) == pytest.approx(-math.pi / 2.0)
    assert wrap_angle(-3.0 * math.pi / 2.0) == pytest.approx(math.pi / 2.0)
    assert wrap_angle(math.pi) == pytest.approx(math.pi)


def test_quaternion_yaw_reads_the_heading():
    """The yaw must come from the pose AMCL and Gazebo both publish."""
    assert quaternion_yaw((0.0, 0.0, 0.0, 1.0)) == pytest.approx(0.0)
    quarter = math.pi / 4.0
    assert quaternion_yaw(
        (0.0, 0.0, math.sin(quarter), math.cos(quarter))) == pytest.approx(
            math.pi / 2.0)
    assert quaternion_yaw((0.0, 0.0, 1.0, 0.0)) == pytest.approx(math.pi)


def test_odom_yaw_reads_the_odometry_pose():
    """turn_by() measures its rotation on /odom, so this must be exact."""
    odometry = Odometry()
    quarter = math.pi / 4.0
    odometry.pose.pose.orientation.z = math.sin(quarter)
    odometry.pose.pose.orientation.w = math.cos(quarter)
    assert odom_yaw(odometry) == pytest.approx(math.pi / 2.0)


def test_turn_twist_only_spins_the_base():
    """The search turns in place; a forward component would drive away."""
    command = make_turn_twist(0.5)
    assert command.linear.x == 0.0
    assert command.linear.y == 0.0
    assert command.angular.z == pytest.approx(0.5)
    assert make_turn_twist(-0.5).angular.z == pytest.approx(-0.5)


def test_search_camera_points_forward_and_down():
    """
    The camera looks along the hand +z axis.

    At this pose it must look ahead of the base (not at the floor beneath
    it), otherwise the 360 degree search only scans a 0.3 m spot. The tilt
    keeps the cube's top face in view at 0.5-3 m.
    """
    rotation = quaternion_matrix(search_camera_quaternion())
    direction = rotation @ np.array([0.0, 0.0, 1.0])
    tilt = SEARCH_CAMERA_TILT
    assert direction[0] == pytest.approx(math.cos(tilt), abs=1e-9)
    assert direction[1] == pytest.approx(0.0, abs=1e-9)
    assert direction[2] == pytest.approx(-math.sin(tilt), abs=1e-9)
    # Mostly forward, slightly down.
    assert direction[0] > 0.9
    assert -0.5 < direction[2] < 0.0


def test_a_full_turn_is_covered_in_steps():
    """Twelve 30 degree steps are a full circle."""
    assert SEARCH_STEP_ANGLE * SEARCH_STEPS == pytest.approx(2.0 * math.pi)
    assert SEARCH_STEPS == 12


def test_standoff_goal_stops_short_of_the_cube_and_faces_it():
    """The goal is on the robot's side, at the standoff distance."""
    cube = (2.0, 0.0)
    robot = (0.0, 0.0)
    x, y, yaw = standoff_goal(cube, robot, 0.5)
    assert math.hypot(x - cube[0], y - cube[1]) == pytest.approx(0.5)
    assert (x, y) == pytest.approx((1.5, 0.0))
    assert yaw == pytest.approx(0.0)

    cube = (0.0, 2.0)
    robot = (0.0, 0.0)
    x, y, yaw = standoff_goal(cube, robot, 0.5)
    assert (x, y) == pytest.approx((0.0, 1.5))
    assert yaw == pytest.approx(math.pi / 2.0)
    # The goal faces the cube: its +x axis points at the cube centre.
    assert math.cos(yaw) == pytest.approx(0.0, abs=1e-9)


def test_standoff_goal_keeps_the_approach_side():
    """Approaching from +y must not send the robot around the cube."""
    x, y, yaw = standoff_goal((1.0, 1.0), (1.0, 3.0), 0.5)
    assert (x, y) == pytest.approx((1.0, 1.5))
    assert yaw == pytest.approx(-math.pi / 2.0)


def test_standoff_goal_handles_a_robot_on_the_cube():
    """A zero separation must still produce a valid goal, not a divide by 0."""
    x, y, yaw = standoff_goal((1.0, 1.0), (1.0, 1.0), 0.5)
    assert math.hypot(x - 1.0, y - 1.0) == pytest.approx(0.5)
    assert math.isfinite(yaw)


def test_standoff_is_within_the_arm_reach():
    """
    One metre is beyond the Panda's 0.85 m reach from panda_link0.

    The visual grasp measures the cube from the look pose (0.58 m ahead of
    the arm base) and expects it 0.45-0.55 m in front of the base, so the
    default standoff must stay in that band.
    """
    assert 0.4 <= STANDOFF_DISTANCE <= 0.6


def test_navigate_goal_carries_the_map_pose_and_heading():
    """Nav2 needs a frame, a position and a yaw quaternion."""
    goal = make_navigate_goal(1.5, -0.25, math.pi / 2.0)
    assert goal.pose.header.frame_id == MAP_FRAME == 'map'
    assert goal.pose.pose.position.x == pytest.approx(1.5)
    assert goal.pose.pose.position.y == pytest.approx(-0.25)
    orientation = goal.pose.pose.orientation
    assert quaternion_yaw(
        (orientation.x, orientation.y, orientation.z,
         orientation.w)) == pytest.approx(math.pi / 2.0)
    qx, qy, qz, qw = yaw_quaternion(0.7)
    assert (qz, qw) == pytest.approx((math.sin(0.35), math.cos(0.35)))


def test_retry_backs_off_before_the_second_attempt():
    """The failure recovery in the request: 20 cm back, then try again."""
    assert GRASP_ATTEMPTS >= 2
    assert RETREAT_DISTANCE == pytest.approx(0.2)


def test_node_reuses_the_visual_grasp_sequence():
    """The grasp stage must be the proven visual_grasp code, not a copy."""
    assert issubclass(SearchAndGraspNode, VisualGraspNode)
    assert hasattr(VisualGraspNode, 'grasp_after_arrival')
    with open(os.path.join(
            PKG_ROOT, 'my_robot_sim', 'visual_grasp.py'),
            encoding='utf-8') as handle:
        source = handle.read()
    assert 'return self.grasp_after_arrival()' in source
    # The orchestrated grasp must not drive forward: Nav2 already placed the
    # base, and the fixed 1 m drive would push the robot into the cube.
    grasp = source[source.index('def grasp_after_arrival'):]
    assert 'self.drive_forward' not in grasp


def test_run_order_is_search_then_navigate_then_grasp():
    """The state machine must not navigate before it knows where the cube is."""
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    run = source[source.index('    def run(self):'):]
    search = run.index('self.search_for_cube()')
    navigate = run.index('self.navigate_to_cube(cube_xy)')
    grasp = run.index('self.grasp_after_arrival()')
    retreat = run.index('self.retreat()')
    assert search < navigate < grasp
    # The retry happens after the first grasp attempt, in the same loop, and
    # re-approaches the cube before grasping again: 0.7 m away the wrist
    # camera is too far to measure the cube reliably.
    assert grasp < retreat
    assert 'self.navigate_to_cube(cube_xy)' in run[retreat:]


def test_search_uses_the_camera_not_the_lidar():
    """
    The scan plane is at z ~= 0.25 m and only grazes the 5 cm cube.

    Even when a ray hits it, a scan cannot tell the red cube from the
    pedestal, the walls or the other boxes, so the search must rely on the
    wrist camera and must not subscribe to /scan.
    """
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    assert 'LaserScan' not in source
    assert "'/scan'" not in source
    assert 'detect_once' in source or 'wait_for_detection' in source
