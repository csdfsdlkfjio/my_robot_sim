"""Unit tests for the search-and-grasp orchestration helpers."""

import math
import os
import types

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry

from my_robot_sim import search_and_grasp as search_module
from my_robot_sim.mobile_manipulation_node import point_in_world
from my_robot_sim.search_and_grasp import (
    GRASP_ATTEMPTS,
    GRASP_LATERAL_MAX,
    GRASP_REACH_MAX,
    GRASP_REACH_MIN,
    MAP_FRAME,
    NAV_GOAL_CLEARANCE,
    NAV_NEAR_GOAL_DISTANCE,
    NAV_NEAR_GOAL_TIMEOUT,
    RETREAT_DISTANCE,
    SEARCH_CAMERA_TILT,
    SEARCH_STEP_ANGLE,
    SEARCH_STEPS,
    STANDOFF_DISTANCE,
    SearchAndGraspNode,
    clearance_violation,
    fallback_look_position,
    make_navigate_goal,
    make_turn_twist,
    map_cell,
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
    the arm base) and expects it 0.45-0.65 m in front of the base, so the
    default standoff must stay in that band. It also has to clear the long
    table; test_map_assets checks that separately.
    """
    assert 0.55 <= STANDOFF_DISTANCE <= 0.65


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
    # Retreating with the arm still down at the cube drags the gripper
    # through it and knocks the cube over (seen live: the cube flew 2 m and
    # the retry's bogus measurement tipped the robot). Fold the arm first.
    retry = run[run.index('抓取失败，后退'):]
    assert retry.index('self.send_home_trajectory()') < retry.index(
        'self.retreat()')


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


def test_map_cell_converts_world_points_to_grid_indices():
    """map_server's origin is the lower-left corner of the bottom-left cell."""
    assert map_cell(0.0, 0.0, -4.35, -4.35, 0.05, 174, 174) == (87, 87)
    assert map_cell(3.95, 0.0, -4.35, -4.35, 0.05, 174, 174) == (166, 87)
    assert map_cell(-4.35, -4.35, -4.35, -4.35, 0.05, 174, 174) == (0, 0)


def test_map_cell_rejects_points_outside_the_map():
    """A goal off the map must be rejected, not indexed into the grid."""
    assert map_cell(4.40, 0.0, -4.35, -4.35, 0.05, 174, 174) is None
    assert map_cell(0.0, -4.40, -4.35, -4.35, 0.05, 174, 174) is None
    assert map_cell(0.0, 0.0, -4.35, -4.35, 0.0, 174, 174) is None


def test_clearance_violation_rejects_walls_obstacles_and_unknown():
    """The standoff must keep the robot's foot print out of everything solid."""
    width = height = 20
    free = [0] * (width * height)
    assert clearance_violation(free, width, height, 10, 10, 2) is None
    wall = list(free)
    wall[12 * width + 12] = 100
    assert clearance_violation(wall, width, height, 10, 10, 2) is not None
    unknown = list(free)
    unknown[8 * width + 8] = -1
    assert clearance_violation(unknown, width, height, 10, 10, 2) is not None
    # One cell from the edge: the 2-cell margin leaves the grid.
    assert clearance_violation(free, width, height, 1, 1, 2) is not None


def test_fallback_look_position_aims_above_a_reachable_cube():
    """An off-standoff stop must still put the camera over the cube."""
    assert fallback_look_position((0.58, 0.0), 0.42) == pytest.approx(
        (0.58, 0.0, 0.42))
    assert fallback_look_position((0.34, 0.12), 0.42) == pytest.approx(
        (0.34, 0.12, 0.42))
    assert fallback_look_position((0.68, -0.10), 0.42) == pytest.approx(
        (0.68, -0.10, 0.42))


def test_fallback_look_position_rejects_an_unreachable_cube():
    """A cube behind or beyond the arm must fail, not get a fake pose."""
    assert GRASP_REACH_MIN < 0.58 < GRASP_REACH_MAX
    too_close = GRASP_REACH_MIN - 0.05
    too_far = GRASP_REACH_MAX + 0.05
    too_sideways = GRASP_LATERAL_MAX + 0.05
    assert fallback_look_position((too_close, 0.0), 0.42) is None
    assert fallback_look_position((too_far, 0.0), 0.42) is None
    assert fallback_look_position((0.5, too_sideways), 0.42) is None
    assert fallback_look_position((0.5, -too_sideways), 0.42) is None


def test_navigation_cancels_and_grasps_after_a_near_goal_timeout():
    """A base circling the goal must not block for the hard 180 s timeout."""
    assert NAV_NEAR_GOAL_DISTANCE >= 0.16
    assert 10.0 <= NAV_NEAR_GOAL_TIMEOUT <= 25.0
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    goal = source[
        source.index('    def send_nav_goal'):
        source.index('    def aim_at_estimated_cube')]
    assert 'NAV_NEAR_GOAL_TIMEOUT' in goal
    assert 'cancel_goal_async' in goal
    assert 'STATUS_CANCELED' in goal
    assert 'STATUS_ABORTED' in goal
    assert "return 'near'" in goal
    # The wait must poll the future (not block) so the clock can fire.
    assert 'spin_once' in goal and 'result_future.done()' in goal
    navigate = source[
        source.index('    def navigate_to_cube'):
        source.index('    def retreat')]
    assert "status == 'near'" in navigate
    assert 'aim_at_estimated_cube' in navigate
    aim = source[
        source.index('    def aim_at_estimated_cube'):
        source.index('    def navigate_to_cube')]
    assert 'fallback_look_position' in aim
    assert 'self.cube_world' in aim
    assert 'self.look_position = look' in aim
    assert 'self.cube_world = point_in_world(' in source


def test_nav_feedback_tracks_the_near_goal_window():
    """The 20 s fallback clock is driven by the action feedback."""
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    feedback = source[
        source.index('    def _on_nav_feedback'):
        source.index('    def send_nav_goal')]
    assert 'distance_remaining' in feedback
    assert 'NAV_NEAR_GOAL_DISTANCE' in feedback
    assert 'nav_near_since' in feedback


class FakeFuture:
    """Minimal rclpy Future stand-in for the navigation wait loop."""

    def __init__(self):
        self._done = False
        self._result = None

    def done(self):
        return self._done

    def result(self):
        return self._result


class FakeLogger:
    """Swallow the node's log calls."""

    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class FakeNavHandle:
    """Accepted action goal with a controllable result future."""

    def __init__(self, result_future):
        self.accepted = True
        self.result_future = result_future
        self.canceled = False

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.canceled = True
        return FakeFuture()


class FakeNavClient:
    """Action client that accepts immediately and records the callback."""

    def __init__(self, handle):
        self.handle = handle
        self.feedback_callback = None

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.feedback_callback = feedback_callback
        future = FakeFuture()
        future._done = True
        future._result = self.handle
        return future


def make_nav_node(handle):
    """Build a SearchAndGraspNode with only the state send_nav_goal touches."""
    node = object.__new__(SearchAndGraspNode)
    node.nav_client = FakeNavClient(handle)
    node.get_logger = lambda: FakeLogger()
    node.stop_and_settle = lambda: None
    node.last_nav_feedback = 0.0
    node.nav_distance_remaining = None
    node.nav_near_since = None
    return node


def install_fake_spin(monkeypatch, node, clock, distances, tick=1.0):
    """Drive the wait loop: advance time and publish a feedback distance."""
    state = {'index': 0}

    def spin_once(owner, timeout_sec=0.0):
        clock[0] += tick
        index = min(state['index'], len(distances) - 1)
        state['index'] += 1
        feedback = types.SimpleNamespace(
            feedback=types.SimpleNamespace(
                distance_remaining=distances[index]))
        node.nav_client.feedback_callback(feedback)

    def spin_until(owner, future, timeout_sec=0.0):
        future._done = True

    monkeypatch.setattr(search_module.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(search_module.rclpy, 'spin_once', spin_once)
    monkeypatch.setattr(
        search_module.rclpy, 'spin_until_future_complete', spin_until)
    monkeypatch.setattr(
        search_module.time, 'monotonic', lambda: clock[0])


def test_send_nav_goal_returns_near_after_the_endgame_timeout(monkeypatch):
    """0-0.16 m oscillation must cancel after 20 s, not wait 180 s."""
    clock = [1000.0]
    handle = FakeNavHandle(FakeFuture())
    node = make_nav_node(handle)
    install_fake_spin(monkeypatch, node, clock, [0.10])

    outcome = node.send_nav_goal(1.0, 0.0, 0.0, timeout=100.0)

    assert outcome == 'near'
    assert handle.canceled
    # One tick to deliver the first feedback, then the 20 s window.
    assert clock[0] - 1000.0 == pytest.approx(NAV_NEAR_GOAL_TIMEOUT + 1.0)


def test_send_nav_goal_keeps_waiting_far_from_the_goal(monkeypatch):
    """A base still metres away must use the hard timeout, not the fallback."""
    clock = [1000.0]
    handle = FakeNavHandle(FakeFuture())
    node = make_nav_node(handle)
    install_fake_spin(monkeypatch, node, clock, [1.5])

    outcome = node.send_nav_goal(1.0, 0.0, 0.0, timeout=5.0)

    assert outcome == 'failed'
    assert handle.canceled
    assert clock[0] - 1000.0 == pytest.approx(5.0)


def test_send_nav_goal_reports_a_canceled_goal(monkeypatch):
    """A CANCELED result must return immediately with a clear failure."""
    clock = [1000.0]
    result = FakeFuture()
    result._done = True
    result._result = types.SimpleNamespace(
        status=GoalStatus.STATUS_CANCELED)
    handle = FakeNavHandle(result)
    node = make_nav_node(handle)
    install_fake_spin(monkeypatch, node, clock, [0.0])

    assert node.send_nav_goal(1.0, 0.0, 0.0) == 'failed'
    assert not handle.canceled
    assert clock[0] == pytest.approx(1000.0)


def make_base_pose(x):
    """Ground-truth base pose at x with an identity rotation."""
    return types.SimpleNamespace(
        position=types.SimpleNamespace(x=x, y=0.0, z=0.2),
        orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))


def test_fallback_aims_with_the_map_independent_world_point():
    """
    AMCL drift is what stopped Nav2 short, so it must not aim the camera.

    The cube was measured in the arm frame at search time; that point is
    stored in the ground-truth world frame and re-expressed at the current
    base pose. Here the base moved 10 cm back, so the cube is 10 cm further
    ahead than it was during the search.
    """
    node = object.__new__(SearchAndGraspNode)
    node.look_position = (0.58, 0.0, 0.42)
    node.get_logger = lambda: FakeLogger()
    node.cube_world = point_in_world(
        (0.50, 0.0, 0.04), make_base_pose(1.0))
    node.base_pose = lambda: make_base_pose(0.90)

    assert node.aim_at_estimated_cube((0.0, 0.0))
    assert node.look_position[0] == pytest.approx(0.60)
    assert node.look_position[1] == pytest.approx(0.0)
    assert node.look_position[2] == pytest.approx(0.42)


def test_navigation_refuses_a_goal_outside_the_free_map():
    """
    The standoff comes from a vision estimate and can land in a wall.

    navigate_to_cube must validate it against map_server's grid before the
    goal is sent, otherwise a bad estimate drives the base into the wall.
    """
    assert NAV_GOAL_CLEARANCE > 0.0
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    navigate = source[
        source.index('    def navigate_to_cube'):
        source.index('    def retreat')]
    assert 'self.wait_for_occupancy_map()' in navigate
    assert 'goal_is_free' in navigate
    assert navigate.index('goal_is_free') < navigate.index('send_nav_goal')
