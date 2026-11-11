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
    FINISH_DISTANCE_LIMIT,
    FINISH_SPEED,
    FINISH_TURN_TIMEOUT,
    GRASP_ATTEMPTS,
    GRASP_LATERAL_MAX,
    GRASP_REACH_MAX,
    GRASP_REACH_MIN,
    MAP_FRAME,
    NAV_GOAL_BEARING_OFFSETS,
    NAV_GOAL_CLEARANCE,
    NAV_GOAL_RETREAT_STEPS,
    NAV_GOAL_SAFE_CLEARANCE,
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
    standoff_candidates,
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


def test_standoff_goal_bearing_offset_moves_the_goal_but_faces_the_cube():
    """
    A different approach bearing keeps the arm's reach and the camera aim.

    The radius stays at the standoff, so the cube is as far away as the
    generic geometry expects, and the goal yaw still points the base at it.
    """
    cube = (1.45, 0.6)
    robot = (0.0, 0.0)
    nominal = standoff_goal(cube, robot, 0.6)
    for offset in (math.radians(15.0), -math.radians(30.0)):
        x, y, yaw = standoff_goal(cube, robot, 0.6, offset)
        assert math.hypot(x - cube[0], y - cube[1]) == pytest.approx(0.6)
        assert yaw == pytest.approx(
            math.atan2(cube[1] - y, cube[0] - x))
        assert math.hypot(x - nominal[0], y - nominal[1]) > 0.05


def test_standoff_candidates_try_the_nominal_goal_first():
    """
    Preferred order: nominal, every bearing at the standoff, then retreats.

    A retreat lengthens the cube-to-base distance, and panda_link0 measures
    about 4 cm more than base_link; at y=+0.6 m the retreat was chosen over
    an equally free bearing and the base ended 0.74 m from the cube, past the
    0.70 m reach limit. Reach-preserving bearings must be tried first, and no
    retreat may go beyond 5 cm.
    """
    cube = (1.45, 0.6)
    robot = (0.48, -0.14)
    candidates = list(standoff_candidates(cube, robot, 0.6))
    offset, extra, x, y, yaw = candidates[0]
    assert (offset, extra) == (0.0, 0.0)
    assert (x, y, yaw) == pytest.approx(standoff_goal(cube, robot, 0.6))
    # All bearings at the nominal radius come before the first retreat.
    first_retreat = next(i for i, c in enumerate(candidates) if c[1] > 0.0)
    assert first_retreat == len(NAV_GOAL_BEARING_OFFSETS)
    assert candidates[1][1] == 0.0 and candidates[1][0] != 0.0
    assert max(NAV_GOAL_RETREAT_STEPS) <= 0.05
    for offset, extra, x, y, yaw in candidates:
        # Every candidate keeps the cube within the arm's reach.
        assert math.hypot(x - cube[0], y - cube[1]) == pytest.approx(
            0.6 + extra)
        assert math.hypot(x - cube[0], y - cube[1]) <= GRASP_REACH_MAX + 1e-9
        assert yaw == pytest.approx(
            math.atan2(cube[1] - y, cube[0] - x))


def test_a_shallow_approach_bearing_finds_a_goal_clear_of_the_table():
    """
    Left beside the table, the direct standoff line crosses the table band.

    Measured live: the base sat at (0.48, -0.14) with the cube at
    (1.48, 0.60); the nominal goal (1.00, 0.24) is only 15 cm from the
    table's front face (needs 24 cm), and 5/10 cm retreats were also
    refused. Turning the approach bearing about the cube moves the goal off
    the table while keeping the standoff radius.
    """
    cube = (1.48, 0.60)
    robot = (0.48, -0.14)
    resolution = 0.05
    origin_x, origin_y = -1.0, -1.0
    width, height = 80, 60
    data = [0] * (width * height)
    # Table strip: x in [1.15, 1.75], y in [-1.0, 1.0].
    col0 = int(round((1.15 - origin_x) / resolution))
    col1 = int(round((1.75 - origin_x) / resolution))
    row0 = int(round((-1.0 - origin_y) / resolution))
    row1 = int(round((1.0 - origin_y) / resolution))
    for row in range(row0, row1):
        for col in range(col0, col1):
            data[row * width + col] = 100

    def free(x, y):
        return clearance_violation(
            data, width, height, origin_x, origin_y, resolution,
            x, y, NAV_GOAL_CLEARANCE) is None

    nominal = standoff_goal(cube, robot, STANDOFF_DISTANCE)
    assert not free(nominal[0], nominal[1])
    free_candidates = [
        candidate for candidate in
        standoff_candidates(cube, robot, STANDOFF_DISTANCE)
        if free(candidate[2], candidate[3])]
    assert free_candidates, 'no bearing clears the table'
    assert any(abs(candidate[0]) > 1e-6 for candidate in free_candidates)


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
    resolution = 0.1
    free = [0] * (width * height)
    assert clearance_violation(
        free, width, height, 0.0, 0.0, resolution, 1.05, 1.05, 0.2) is None
    # Cell (12, 10) covers x 1.2-1.3, y 1.0-1.1: 0.15 m from the goal.
    wall = list(free)
    wall[10 * width + 12] = 100
    assert clearance_violation(
        wall, width, height, 0.0, 0.0, resolution,
        1.05, 1.05, 0.2) == '附近有墙壁或障碍物'
    unknown = list(free)
    unknown[8 * width + 8] = -1
    assert clearance_violation(
        unknown, width, height, 0.0, 0.0, resolution,
        1.05, 1.05, 0.2) == '附近是未知区域'
    # 0.15 m from the same wall: a 0.14 m clearance fits, a 0.2 m one does
    # not. Measuring to the cell rectangle (not its centre) is what lets
    # lateral goals next to the table pass.
    assert clearance_violation(
        wall, width, height, 0.0, 0.0, resolution,
        1.05, 1.05, 0.14) is None
    # The clearance disc leaves the grid at the lower-left corner.
    assert clearance_violation(
        free, width, height, 0.0, 0.0, resolution,
        0.15, 0.15, 0.2) == '离地图边界不足'


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


def test_navigation_drives_the_last_stretch_before_the_fallback_grasp():
    """
    Nav2 can stall a few centimetres short, which leaves the cube out of reach.

    Live at y=+0.6 m the base stopped 9 cm short of the standoff and 0.74 m
    from the cube (reach limit 0.70 m), so the fallback grasp refused with
    "超出可抓范围". After a 'near' timeout the node must drive the remaining
    bounded stretch before aiming the camera.
    """
    assert 0.0 < FINISH_DISTANCE_LIMIT <= NAV_NEAR_GOAL_DISTANCE + 0.1
    assert 0.0 < FINISH_SPEED <= 0.15
    # 134 deg took ~23 s at the ~0.1 rad/s measured beside the table, so the
    # old 6 s deadline stopped two thirds of the way.
    assert FINISH_TURN_TIMEOUT >= 20.0
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    navigate = source[
        source.index('    def navigate_to_cube'):
        source.index('    def retreat')]
    near = navigate[navigate.index("status == 'near'"):]
    assert 'self.finish_standoff(goal_x, goal_y, yaw)' in near
    assert near.index('finish_standoff') < near.index(
        'aim_at_estimated_cube')
    # A stateful goal checker latches at the edge of the 5 cm xy tolerance
    # and the unbraked base then drifts backwards, so SUCCEEDED can also end
    # 9 cm short. The success path must drive the same bounded remainder.
    after_near = navigate[navigate.index("status == 'near'"):]
    branch_end = after_near.index('return True') + len('return True')
    success = after_near[branch_end:]
    assert 'self.finish_standoff(goal_x, goal_y, yaw)' in success, (
        'a latched-but-drifted success leaves the cube out of arm reach')
    assert success.index('finish_standoff') < success.index(
        'aim_at_estimated_cube')
    finish = source[
        source.index('    def map_pose_in_odom'):
        source.index('    def aim_at_estimated_cube')]
    assert 'FINISH_DISTANCE_LIMIT' in finish
    assert 'make_forward_twist(FINISH_SPEED)' in finish
    assert 'odom_distance' in finish
    assert 'stop_and_settle' in finish
    # Stop before measuring the turn: leftover Nav2 commands would fight it.
    assert finish.index('stop_and_settle') < finish.index('turn_by')
    assert 'ODOM_FRAME, MAP_FRAME' in finish


def test_navigation_reaims_the_camera_after_a_successful_goal():
    """
    A successful Nav2 goal can still leave the cube out of the camera.

    Nav2 allows 0.25 rad (14 deg) of yaw error. The wrist camera sits about
    14 cm above the cube top, so its 60 deg view only covers +/-8 cm; at the
    0.6 m standoff a 9 deg heading error moves the cube
    0.6 * sin(9 deg) = 9.4 cm sideways, already out of frame. The fixed
    look pose therefore missed the y=+0.6 m cube for the whole 10 s search
    even though navigation had reported SUCCEEDED. navigate_to_cube must
    re-aim at the cube on the success path too, not only after the 'near'
    timeout.
    """
    assert STANDOFF_DISTANCE * math.sin(math.radians(9.0)) > 0.08
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    navigate = source[
        source.index('    def navigate_to_cube'):
        source.index('    def retreat')]
    after_near = navigate[navigate.index("status == 'near'"):]
    branch_end = after_near.index('return True') + len('return True')
    success = after_near[branch_end:]
    assert 'self.stop_and_settle()' in success, success
    assert 'aim_at_estimated_cube' in success, (
        'the success path leaves the fixed look pose pointing at empty table')


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


def test_navigation_prefers_goals_the_local_costmap_can_reach():
    """
    A goal inside the table's inscribed band wedges DWB.

    Live at y=+0.6 m the goal (0.876, 0.548) passed the 0.24 m map check but
    its costmap cell was 99 = inscribed obstacle; DWB logged "No valid
    trajectories out of 419, BaseObstacle hits obstacle" and the base sat
    0.28 m short until Nav2 aborted. Goals must first be sought with one
    more cell of clearance, and only then at the bare physical minimum.
    """
    assert NAV_GOAL_CLEARANCE < NAV_GOAL_SAFE_CLEARANCE <= 0.32
    with open(MODULE, encoding='utf-8') as handle:
        source = handle.read()
    assert ('def goal_is_free(self, x, y, clearance=NAV_GOAL_CLEARANCE)'
            in source)
    navigate = source[
        source.index('    def navigate_to_cube'):
        source.index('    def retreat')]
    assert 'first_free(NAV_GOAL_SAFE_CLEARANCE)' in navigate
    assert (navigate.index('first_free(NAV_GOAL_SAFE_CLEARANCE)')
            < navigate.index('first_free(NAV_GOAL_CLEARANCE)'))
    # The nominal goal can pass the bare 0.24 m check and still be replaced
    # for failing the 0.30 m one; the bare check returns no reason then, and
    # run9 logged "导航目标 (0.87, 0.36) 不可用：；..." with a blank reason.
    # The strict reason must be captured before the candidates are tried.
    assert 'strict_reason' in navigate
    assert navigate.index('strict_reason') < navigate.index('def first_free')
