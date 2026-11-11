"""
Search for the red cube, drive to it with Nav2, then grasp it.

Stages:
    1. search   - aim the wrist camera forward and turn the base in place a
                  full turn, stopping at the first red blob;
    2. navigate - send the cube's map position minus a standoff distance to
                  Nav2's NavigateToPose action and wait for the base. The
                  goal is checked against map_server's grid first and refused
                  when it falls in a wall, an obstacle or off the map. If the
                  base circles the goal for 20 s without SUCCEEDED, cancel
                  and re-aim the wrist camera at the cube estimate;
    3. grasp    - run the visual-grasp sequence (VisualGraspNode) on the
                  settled base;
    4. retry    - fold the arm home, back away 20 cm, drive back to the
                  standoff and run the grasp once more if it failed. Folding
                  first keeps the failed grasp's open gripper from dragging
                  the cube away as the base reverses.

Why the camera and not the lidar: lidar_link sits 5 cm above base_link, which
is 0.20 m high, so the scan plane is at z ~= 0.25 m -- it grazes the cube
(0.225-0.275 m) and only a couple of rays can hit it. A scan can also never
identify the cube as red: the table, the walls and the blue/orange boxes
all come back as obstacles. The wrist camera is the sensor that both sees the
small cube and knows its colour.

Usage:
    # mobile_manipulation.launch.py defaults to the enclosed demo.world room
    ros2 launch my_robot_sim mobile_manipulation.launch.py
    ros2 launch my_robot_sim nav2.launch.py
    # config/nav2_params.yaml already gives AMCL the map's initial pose.
    # If the robot has moved, pass initial_x / initial_y / initial_yaw with
    # publish_initial_pose:=true, or set it in RViz with 2D Pose Estimate.
    ros2 run my_robot_sim search_and_grasp

maps/my_map.pgm is generated from demo.world's geometry in the world frame,
so the robot's spawn pose is (0, 0, 0) in the map and that is the initial pose
in config/nav2_params.yaml. The node still never converts ground-truth world
coordinates into map goals: everything that goes to Nav2 is expressed through
the map -> base_link TF that AMCL publishes.
"""

import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    Twist,
)
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformException

from my_robot_sim.mobile_manipulation_node import (
    ARM_BASE_FRAME,
    GRIPPER_OPEN,
    ODOM_FRAME,
    TF_TIMEOUT,
    make_forward_twist,
    make_gripper_goal,
    make_reach_goal,
    odom_distance,
    odom_xy,
    point_in_world,
)
from my_robot_sim.visual_grasp import (
    VisualGraspNode,
    source_name,
    transform_point,
)

MAP_FRAME = 'map'
ROBOT_FRAME = 'base_link'
NAV_ACTION = 'navigate_to_pose'

# Search pose. The wrist camera looks along the hand +z axis, so rotating the
# hand by 90 + tilt degrees about y points it forward and ``tilt`` degrees
# down. At this height the cube's top face stays in view from about 0.5 m to
# 3 m, which covers everything the base can see from one spot.
SEARCH_HAND_POSITION = (0.40, 0.0, 0.48)
SEARCH_CAMERA_TILT = math.radians(20.0)
# 12 steps of 30 degrees cover a full turn. The first step looks from the
# current heading, so a cube straight ahead is found without turning.
SEARCH_STEP_ANGLE = math.pi / 6.0
SEARCH_STEPS = 12
SEARCH_TURN_SPEED = 0.5  # rad/s
SEARCH_STEP_TIMEOUT = 10.0  # s per 30 degree step
SEARCH_SETTLE = 0.6  # s after each step, for a fresh camera frame
SEARCH_DETECTION_TIMEOUT = 1.0  # s per step
SEARCH_SETTLE_AFTER_LOOK = 0.3  # s, lets the arm stop before the first look

# Nav2 goal distance from the cube centre, squeezed between the arm and the
# table's local costmap:
#   * reach: the grasp refuses beyond GRASP_REACH_MAX = 0.70 m in panda_link0,
#     which sits 4 cm behind base_link, so the base may be at most ~0.66 m
#     from the cube (0.70 m read 0.74 m and was refused live);
#   * costmap: the table's front face is x=1.15 and the cube x=1.455. The
#     local costmap marks the whole band x >= 0.875 (cells at 0.90 and beyond
#     cost 99 = inscribed obstacle) because the robot radius 0.22 plus one
#     cell. A goal there makes DWB report "No valid trajectories out of 419,
#     BaseObstacle hits obstacle", the base wedges 0.28 m short and Nav2
#     aborts. 0.63 m puts the goal at x ~ 0.84, still reachable (x >= 0.795)
#     and outside the inscribed band.
# The visual grasp expects the cube 0.45-0.65 m in front of base_link, so
# 0.63 also stays inside the band it was tuned for.
STANDOFF_DISTANCE = 0.63  # m
NAV_START_TIMEOUT = 30.0  # s for the goal to be accepted
NAV_TIMEOUT = 180.0  # s hard cap for the goal to finish
NAV_FEEDBACK_PERIOD = 3.0  # s between distance logs
# Nav2 can circle the last few centimetres without ever reporting SUCCEEDED:
# the feedback only exposes the translational error, so a base sitting at
# "0.00 m remaining" but not yaw-aligned looks stuck. When the error has
# stayed inside NAV_NEAR_GOAL_DISTANCE for NAV_NEAR_GOAL_TIMEOUT seconds,
# cancel the goal, stop, and let the wrist camera finish from there.
NAV_NEAR_GOAL_DISTANCE = 0.20  # m
NAV_NEAR_GOAL_TIMEOUT = 20.0  # s
# The fallback grasp aims the wrist camera at the cube's estimated position
# instead of the fixed look pose, so the base does not have to stop exactly
# on the standoff. These bounds keep that aim inside the Panda's reach from
# panda_link0 (0.85 m) with room for the approach.
GRASP_REACH_MIN = 0.30  # m ahead of the arm base
GRASP_REACH_MAX = 0.70  # m
GRASP_LATERAL_MAX = 0.25  # m to either side

# The standoff point comes from a vision estimate. If that estimate is wrong
# the point can fall outside the room, and Nav2 would plan into the wall or
# off the map. Check the goal against map_server's own grid before sending
# it: the goal point must be at least this far from every occupied cell.
MAP_TOPIC = '/map'
# 0.22 m is the footprint radius used by both costmaps; 2 cm on top absorbs
# the vision estimate's error. The check measures the distance from the goal
# point to the actual cell rectangles, not to whole-cell centres: the older
# square-of-cells test over-rejected diagonal goals next to the long table by
# up to a cell diagonal, which made a cube at y=+/-0.6 m unusable even though
# the base would have kept 25 cm of real clearance.
NAV_GOAL_CLEARANCE = 0.24  # m; the robot's footprint radius plus a margin
# Nav2's local costmap inflates the table by the 0.22 m robot radius, and a
# goal whose cell lands inside that inscribed band is unreachable no matter
# what DWB does: every trajectory reports "BaseObstacle hits obstacle" and
# the base wedges (observed 0.28 m short, then ABORTED). The band's edge sits
# at x ~ 0.875 in this world (cells at 0.90 and east cost 99), so goals need
# roughly one more cell of slack than the footprint: 0.30 m puts them at
# x <= 0.85. Candidates are tried at this clearance first and only then at
# NAV_GOAL_CLEARANCE, so a biased cube estimate can still be approached.
NAV_GOAL_SAFE_CLEARANCE = 0.30  # m; keeps the goal cell out of the band
OCCUPIED_THRESHOLD = 65  # percent, equal to my_map.yaml's occupied_thresh
# Extra standoff distances tried when no bearing at the nominal standoff
# clears the table. A cube at |y| = 0.6 m seen obliquely can be estimated
# ~10 cm too far east; a small retreat moves the goal west of the table
# while keeping the true cube within the arm's reach.
# 2 cm is the most the arm allows on top of 0.63: the radius is measured
# from the cube, while the grasp measures from panda_link0, which sits
# ARM_MOUNT_OFFSET (4 cm) behind base_link. 0.70 m at most keeps the
# measured distance under GRASP_REACH_MAX = 0.70 m with a little slack.
NAV_GOAL_RETREAT_STEPS = (0.02,)
# Retreating along the cube->base line only helps when that line is roughly
# perpendicular to the long table. If the base was left at a shallow angle
# (e.g. beside the table after the previous run), every point on the line
# inside the arm's reach lies in the table's clearance band -- measured from
# (0.48, -0.14) the nominal goal (1.00, 0.24) was only 15 cm from the table
# face, and 5/10 cm of retreat still left it under 24 cm. Try neighbouring
# bearings around the cube instead: the radius stays at the standoff, so the
# arm still reaches, while a different approach direction moves the goal off
# the table.
NAV_GOAL_BEARING_OFFSETS = (
    0.0,
    math.radians(15.0), -math.radians(15.0),
    math.radians(30.0), -math.radians(30.0),
    math.radians(45.0), -math.radians(45.0))

# Nav2's controller can give up within NAV_NEAR_GOAL_DISTANCE of the goal
# (it stalls during the final in-place rotation next to the inflated table).
# The last stretch is then driven directly: the goal already passed the map
# clearance check, /odom is Gazebo ground truth, and the motion is bounded.
# Live at y=+0.6 m the base settled 9 cm short and 0.74 m from the cube,
# outside the arm's reach, so the fallback grasp could not start.
FINISH_SPEED = 0.08  # m/s
FINISH_DISTANCE_LIMIT = 0.25  # m; farther than this is not a last stretch
FINISH_TOLERANCE = 0.02  # m short of the goal is fine
FINISH_TIMEOUT = 8.0  # s for the direct drive
FINISH_YAW_TOLERANCE = math.radians(4.0)
# 6 s was too short for the last rotation. Commanded 0.5 rad/s the base turns
# at ~0.39 rad/s on open ground, but next to the table it has been measured
# at ~0.1 rad/s (the driven wheels slip and the unpowered caster drags), so a
# 134 deg correction needed ~23 s and every 6 s attempt stopped two thirds
# of the way with "只转了 ...". turn_by leaves the loop as soon as the target
# is reached, so a generous deadline only costs time when the base really is
# stuck.
FINISH_TURN_TIMEOUT = 25.0  # s per rotation

# AMCL has to publish map -> base_link before a map-frame goal means anything.
MAP_WAIT_TIMEOUT = 30.0  # s
INITIAL_POSE_COVARIANCE = 0.25  # m^2 / rad^2, a deliberately loose prior

# Retry: back off and let the wrist camera see the cube from further away.
GRASP_ATTEMPTS = 2
RETREAT_DISTANCE = 0.2  # m
RETREAT_SPEED = 0.12  # m/s
RETREAT_TIMEOUT = 15.0  # s


def wrap_angle(angle):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(quaternion):
    """Yaw of a (x, y, z, w) quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def yaw_quaternion(yaw):
    """Quaternion for a rotation of ``yaw`` about z."""
    return (0.0, 0.0, math.sin(float(yaw) / 2.0),
            math.cos(float(yaw) / 2.0))


def odom_yaw(odometry):
    """Yaw from an Odometry message."""
    orientation = odometry.pose.pose.orientation
    return quaternion_yaw(
        (orientation.x, orientation.y, orientation.z, orientation.w))


def make_turn_twist(speed=SEARCH_TURN_SPEED):
    """Constant spin-in-place velocity command."""
    command = Twist()
    command.linear.x = 0.0
    command.angular.z = float(speed)
    return command


def search_camera_quaternion(tilt=SEARCH_CAMERA_TILT):
    """
    Hand orientation that points the wrist camera forward and ``tilt`` down.

    The camera looks along the hand +z axis. Rotating the hand about y by
    90 + tilt degrees maps +z onto (cos tilt, 0, -sin tilt): forward and down.
    """
    half = (math.pi / 2.0 + float(tilt)) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def standoff_goal(cube_xy, robot_xy, standoff=STANDOFF_DISTANCE,
                  bearing_offset=0.0):
    """
    Return (x, y, yaw) for the Nav2 goal ``standoff`` metres from the cube.

    The goal sits on the line from the cube to the robot, so the base keeps
    the side it already sees the cube from, and it faces the cube.
    ``bearing_offset`` rotates that line about the cube, keeping the radius
    and the facing, to move the goal away from the long table.
    """
    dx = float(robot_xy[0]) - float(cube_xy[0])
    dy = float(robot_xy[1]) - float(cube_xy[1])
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        # Degenerate: the robot thinks it is on top of the cube. Approach
        # from -x, which matches the direction it was driving.
        angle = math.pi
    else:
        angle = math.atan2(dy, dx)
    angle += float(bearing_offset)
    goal_x = float(cube_xy[0]) + float(standoff) * math.cos(angle)
    goal_y = float(cube_xy[1]) + float(standoff) * math.sin(angle)
    yaw = math.atan2(float(cube_xy[1]) - goal_y,
                     float(cube_xy[0]) - goal_x)
    return goal_x, goal_y, yaw


def standoff_candidates(cube_xy, robot_xy, standoff=STANDOFF_DISTANCE):
    """
    Yield (bearing_offset, extra, x, y, yaw) goals, preferred order first.

    The nominal goal comes first, then every neighbouring bearing at the
    nominal standoff, and only then the along-line retreats. Bearing first
    matters: at y=+0.6 m the same bearing plus a 10 cm retreat was the first
    free candidate, but 0.70 m from the cube is past the arm's reach; a
    -15 deg bearing at the unchanged 0.60 m standoff was also free and keeps
    the cube graspable. The caller keeps the first candidate that passes the
    map clearance check, so the reach-preserving ones must come first.
    """
    for extra in (0.0,) + NAV_GOAL_RETREAT_STEPS:
        for offset in NAV_GOAL_BEARING_OFFSETS:
            x, y, yaw = standoff_goal(
                cube_xy, robot_xy, float(standoff) + extra, offset)
            yield offset, extra, x, y, yaw


def fallback_look_position(cube_in_arm, look_z):
    """
    Look pose directly above a cube estimate, or None when it is unreachable.

    Used when Nav2 stopped short of the standoff: the cube can then sit
    several centimetres away from the fixed look pose, and the wrist camera's
    downward view only covers about +/-8 cm there. Aiming at the estimate
    lets the normal visual measurement take over instead of failing blind.
    """
    x = float(cube_in_arm[0])
    y = float(cube_in_arm[1])
    if not GRASP_REACH_MIN <= x <= GRASP_REACH_MAX:
        return None
    if abs(y) > GRASP_LATERAL_MAX:
        return None
    return (x, y, float(look_z))


def map_cell(x, y, origin_x, origin_y, resolution, width, height):
    """
    Return the (column, row) map cell of a point, or None when outside.

    Follows map_server's convention: the origin is the lower-left corner of
    the bottom-left cell and rows are numbered upwards, while the PGM image
    itself is stored top-down.
    """
    resolution = float(resolution)
    if resolution <= 0.0:
        return None
    # The epsilon keeps exact cell boundaries (0.0 with origin -4.35 and a
    # 0.05 m cell) from falling one cell low through binary rounding.
    col = int(math.floor(
        (float(x) - float(origin_x)) / resolution + 1e-9))
    row = int(math.floor(
        (float(y) - float(origin_y)) / resolution + 1e-9))
    if col < 0 or row < 0 or col >= int(width) or row >= int(height):
        return None
    return col, row


def clearance_violation(data, width, height, origin_x, origin_y, resolution,
                        goal_x, goal_y, clearance):
    """
    Reason the point lacks ``clearance`` metres of free space, or None.

    ``data`` is OccupancyGrid.data: row-major from the origin corner, -1 for
    unknown and 0..100 for occupancy. The distance is measured from the goal
    point to each occupied cell's rectangle, so a goal just outside the
    clearance still passes regardless of where it falls inside its own cell.
    Checking whole cells around a rounded goal cell instead over-rejects
    goals next to long obstacles such as the table by up to a cell diagonal.
    """
    width = int(width)
    height = int(height)
    resolution = float(resolution)
    radius = max(0.0, float(clearance))
    if resolution <= 0.0:
        return '地图分辨率无效'
    # Every cell whose rectangle can come within ``radius`` of the goal.
    col_min = int(math.floor(
        (float(goal_x) - radius - float(origin_x)) / resolution + 1e-9))
    col_max = int(math.floor(
        (float(goal_x) + radius - float(origin_x)) / resolution + 1e-9))
    row_min = int(math.floor(
        (float(goal_y) - radius - float(origin_y)) / resolution + 1e-9))
    row_max = int(math.floor(
        (float(goal_y) + radius - float(origin_y)) / resolution + 1e-9))
    if (col_min < 0 or row_min < 0
            or col_max >= width or row_max >= height):
        return '离地图边界不足'
    for row in range(row_min, row_max + 1):
        cell_min_y = float(origin_y) + row * resolution
        cell_max_y = cell_min_y + resolution
        nearest_y = min(max(float(goal_y), cell_min_y), cell_max_y)
        for col in range(col_min, col_max + 1):
            value = data[row * width + col]
            if value < 0:
                return '附近是未知区域'
            if value < OCCUPIED_THRESHOLD:
                continue
            cell_min_x = float(origin_x) + col * resolution
            cell_max_x = cell_min_x + resolution
            nearest_x = min(max(float(goal_x), cell_min_x), cell_max_x)
            if math.hypot(float(goal_x) - nearest_x,
                          float(goal_y) - nearest_y) < radius:
                return '附近有墙壁或障碍物'
    return None


def make_navigate_goal(x, y, yaw, frame_id=MAP_FRAME):
    """
    Nav2 NavigateToPose goal at (x, y, yaw) in ``frame_id``.

    The header stamp is left at zero: tf2 reads a zero stamp as "latest
    available", which avoids mixing this node's clock with Gazebo's sim time.
    """
    goal = NavigateToPose.Goal()
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_quaternion(yaw)
    pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    goal.pose = pose
    return goal


class SearchAndGraspNode(VisualGraspNode):
    """Find the red cube, drive to it with Nav2 and grasp it."""

    def __init__(self):
        super().__init__('search_and_grasp')
        self.standoff = float(self.declare_parameter(
            'standoff', STANDOFF_DISTANCE).value)
        self.search_steps = int(self.declare_parameter(
            'search_steps', SEARCH_STEPS).value)
        self.publish_initial_pose = bool(self.declare_parameter(
            'publish_initial_pose', False).value)
        self.initial_x = float(self.declare_parameter(
            'initial_x', 0.0).value)
        self.initial_y = float(self.declare_parameter(
            'initial_y', 0.0).value)
        self.initial_yaw = float(self.declare_parameter(
            'initial_yaw', 0.0).value)

        self.nav_client = ActionClient(self, NavigateToPose, NAV_ACTION)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)
        self.last_nav_feedback = 0.0
        self.nav_distance_remaining = None
        self.nav_near_since = None
        # Cube's top-face point in the ground-truth world frame. Unlike the
        # map estimate it does not inherit AMCL's drift, so it is the better
        # aim point when Nav2 stopped off the standoff.
        self.cube_world = None
        self.map_transform = None
        self.map_msg = None
        # map_server publishes its grid once, latched; a subscriber that
        # starts later only receives it with transient-local durability.
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            OccupancyGrid, MAP_TOPIC, self._on_map, map_qos)

    # ---------------------------------------------------------------- maps
    def lookup_map_transform(self, timeout=TF_TIMEOUT):
        """Latest map -> panda_link0 transform, or None."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                return self.tf_buffer.lookup_transform(
                    MAP_FRAME, ARM_BASE_FRAME, time_msg())
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def _on_map(self, message):
        """Cache map_server's grid for the navigation goal check."""
        self.map_msg = message

    def wait_for_occupancy_map(self, timeout=MAP_WAIT_TIMEOUT):
        """Wait for map_server's latched grid, not just the map TF."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.map_msg is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.map_msg is not None

    def goal_is_free(self, x, y, clearance=NAV_GOAL_CLEARANCE):
        """Check that (x, y) is free with room for the robot's foot print."""
        if self.map_msg is None:
            return False, '没有收到 %s' % MAP_TOPIC
        info = self.map_msg.info
        reason = clearance_violation(
            self.map_msg.data, info.width, info.height,
            info.origin.position.x, info.origin.position.y,
            info.resolution, x, y, clearance)
        return reason is None, reason or ''

    def wait_for_map(self, timeout=MAP_WAIT_TIMEOUT):
        """
        Wait until AMCL publishes map -> base_link.

        Without it a map-frame goal has no meaning: neither the cube's map
        position nor the base's can be computed.
        """
        if self.publish_initial_pose:
            self.send_initial_pose()
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    MAP_FRAME, ROBOT_FRAME, time_msg())
                return True
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(
            '没有 %s -> %s 的 TF：AMCL 还没有定位。请先在 RViz 用 '
            '"2D Pose Estimate" 给出初始位姿（在地图上点一下并拖出朝向），'
            '或者加参数 publish_initial_pose:=true 并设置 initial_x / '
            'initial_y / initial_yaw。' % (MAP_FRAME, ROBOT_FRAME))
        return False

    def send_initial_pose(self):
        """Publish /initialpose so AMCL starts from a known prior."""
        message = PoseWithCovarianceStamped()
        message.header.frame_id = MAP_FRAME
        message.pose.pose.position.x = self.initial_x
        message.pose.pose.position.y = self.initial_y
        qx, qy, qz, qw = yaw_quaternion(self.initial_yaw)
        message.pose.pose.orientation = Quaternion(
            x=qx, y=qy, z=qz, w=qw)
        covariance = [0.0] * 36
        for index in (0, 7):
            covariance[index] = INITIAL_POSE_COVARIANCE
        covariance[35] = INITIAL_POSE_COVARIANCE
        message.pose.covariance = covariance
        self.get_logger().info(
            '发布初始位姿 (%0.2f, %0.2f, %0.2f rad) 给 AMCL ...'
            % (self.initial_x, self.initial_y, self.initial_yaw))
        self.initial_pose_pub.publish(message)
        self._spin_for(1.0)

    def map_robot_xy(self):
        """Return the base position in the map frame, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(
                MAP_FRAME, ROBOT_FRAME, time_msg())
        except TransformException:
            return None
        translation = transform.transform.translation
        return (translation.x, translation.y)

    # -------------------------------------------------------------- search
    def turn_by(self, angle, timeout=SEARCH_STEP_TIMEOUT):
        """Turn the base in place by ``angle`` radians, measured on /odom."""
        if self.odom is None:
            self.get_logger().error('没有 /odom，无法旋转底盘')
            return False
        start = odom_yaw(self.odom)
        speed = math.copysign(SEARCH_TURN_SPEED, angle)
        deadline = time.monotonic() + timeout
        turned = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            turned = wrap_angle(odom_yaw(self.odom) - start)
            if abs(turned) >= abs(angle):
                break
            self.cmd_pub.publish(make_turn_twist(speed))
            rclpy.spin_once(self, timeout_sec=0.02)
        self._publish_stop(0.3)
        if abs(turned) < abs(angle) - 0.08:
            self.get_logger().warning(
                '只转了 %.1f°（目标 %.1f°），继续搜索'
                % (math.degrees(turned), math.degrees(angle)))
        return True

    def search_for_cube(self):
        """
        Turn the base a full turn and return the cube's map (x, y), or None.

        The first look happens from the current heading, so a cube already in
        view costs no rotation; afterwards the base steps 30 degrees at a
        time. A detection is only trusted when the wrist camera's back
        projection yields a map point, otherwise the search continues.
        """
        orientation = self.orientation_in_arm_frame(
            search_camera_quaternion())
        self.get_logger().info(
            '搜索第一步：把腕部相机移到前视位姿 (%0.2f, %0.2f, %0.2f) '
            '[%s] ...' % (SEARCH_HAND_POSITION[0], SEARCH_HAND_POSITION[1],
                          SEARCH_HAND_POSITION[2], ARM_BASE_FRAME))
        if not self.send_move_goal(
                make_reach_goal(SEARCH_HAND_POSITION, orientation)):
            self.get_logger().error('把相机移到搜索位姿失败')
            return None
        self._spin_for(SEARCH_SETTLE_AFTER_LOOK)

        for step in range(self.search_steps):
            if step > 0:
                self.get_logger().info(
                    '搜索：原地旋转 %.0f°（第 %d/%d 步）...'
                    % (math.degrees(SEARCH_STEP_ANGLE), step + 1,
                       self.search_steps))
                self.turn_by(-SEARCH_STEP_ANGLE)
            self._spin_for(SEARCH_SETTLE)

            detection = self.wait_for_detection(SEARCH_DETECTION_TIMEOUT)
            if detection is None:
                continue
            self.publish_debug_image(detection)
            self.get_logger().info(
                '搜索：第 %d 步看到红色区域，像素 (%0.0f, %0.0f)，'
                '面积 %.0f px²，开始反投影...'
                % (step + 1, detection.u, detection.v, detection.area))
            try:
                point, source = self.target_point_in_arm_frame(detection)
            except RuntimeError as error:
                self.get_logger().warning('红色区域定位失败：%s' % error)
                continue
            horizon = math.hypot(point[0], point[1])
            if horizon > 4.0:
                self.get_logger().warning(
                    '红色区域距离 %.1f m，太远不像方块，继续搜索' % horizon)
                continue

            pose = self.base_pose()
            if pose is not None:
                # Keep a map-independent copy: the fallback grasp aim must
                # not inherit the AMCL error that made Nav2 stop short.
                self.cube_world = point_in_world(point, pose)

            transform = self.lookup_map_transform()
            if transform is None:
                self.get_logger().error('没有 map -> %s 的 TF'
                                        % ARM_BASE_FRAME)
                return None
            map_point = transform_point(transform, point)
            self.get_logger().info(
                '搜索成功：方块在 %s 下为 (%0.4f, %0.4f, %0.4f)（来源：%s），'
                '地图坐标为 (%0.4f, %0.4f)。'
                % (ARM_BASE_FRAME, point[0], point[1], point[2],
                   source_name(source), map_point[0], map_point[1]))
            return (map_point[0], map_point[1])

        self.get_logger().error(
            '搜索失败：转了 %.0f° 也没有找到红色方块。确认方块在视野内，'
            '并且相机话题 %s 有图像。'
            % (math.degrees(SEARCH_STEP_ANGLE * self.search_steps),
               self.image_topic))
        return None

    # ---------------------------------------------------------- navigation
    def _on_nav_feedback(self, feedback):
        """Track the remaining distance and whether the base is at the goal."""
        now = time.monotonic()
        distance = float(feedback.feedback.distance_remaining)
        self.nav_distance_remaining = distance
        if distance <= NAV_NEAR_GOAL_DISTANCE:
            if self.nav_near_since is None:
                self.nav_near_since = now
        else:
            # It left the goal area: the 20 s clock starts over.
            self.nav_near_since = None
        if now - self.last_nav_feedback < NAV_FEEDBACK_PERIOD:
            return
        self.last_nav_feedback = now
        self.get_logger().info('导航中：剩余 %.2f m' % distance)

    def send_nav_goal(self, x, y, yaw, timeout=NAV_TIMEOUT):
        """
        Send one NavigateToPose goal.

        Returns 'succeeded' when Nav2 reached the goal, 'near' when the base
        stayed within NAV_NEAR_GOAL_DISTANCE for NAV_NEAR_GOAL_TIMEOUT
        seconds without SUCCEEDED and the goal was cancelled, and 'failed'
        otherwise. In both non-successful cases the base is already stopped.
        """
        if not self.nav_client.wait_for_server(
                timeout_sec=NAV_START_TIMEOUT):
            self.get_logger().error(
                '找不到 Nav2 的 %s Action。先启动 '
                'ros2 launch my_robot_sim nav2.launch.py' % NAV_ACTION)
            return 'failed'
        goal = make_navigate_goal(x, y, yaw)
        send_future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._on_nav_feedback)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=NAV_START_TIMEOUT)
        if not send_future.done():
            self.get_logger().error('发送导航目标超时')
            return 'failed'
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Nav2 拒绝了导航目标')
            return 'failed'

        self.nav_near_since = None
        self.nav_distance_remaining = None
        result_future = handle.get_result_async()
        deadline = time.monotonic() + float(timeout)
        while rclpy.ok() and not result_future.done():
            now = time.monotonic()
            if now >= deadline:
                self.get_logger().error(
                    '导航 %.0f s 未完成，取消目标' % timeout)
                handle.cancel_goal_async()
                self.stop_and_settle()
                return 'failed'
            if (self.nav_near_since is not None
                    and now - self.nav_near_since
                    >= NAV_NEAR_GOAL_TIMEOUT):
                self.get_logger().warning(
                    '底盘已在目标 %.2f m 内停留 %.0f s 仍未 SUCCEEDED，'
                    '取消导航，改用视觉抓取当前位置 ...'
                    % (NAV_NEAR_GOAL_DISTANCE, NAV_NEAR_GOAL_TIMEOUT))
                handle.cancel_goal_async()
                self.stop_and_settle()
                return 'near'
            rclpy.spin_once(self, timeout_sec=0.1)
        if not result_future.done():
            # rclpy shut down mid-goal; do not leave the base driving.
            handle.cancel_goal_async()
            self.stop_and_settle()
            return 'failed'

        status = result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            return 'succeeded'
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error(
                '导航被取消（CANCELED），底盘没有到达目标')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                '导航被中止（ABORTED），Nav2 无法到达目标')
        else:
            self.get_logger().error('导航失败，Action 状态 %d' % status)
        return 'failed'

    def map_pose_in_odom(self, x, y, yaw, timeout=TF_TIMEOUT):
        """Express a map-frame pose in the odom frame, or return None."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    ODOM_FRAME, MAP_FRAME, time_msg())
                break
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            return None
        point = transform_point(transform, (x, y, 0.0))
        rotation = transform.transform.rotation
        offset_yaw = quaternion_yaw(
            (rotation.x, rotation.y, rotation.z, rotation.w))
        return point[0], point[1], wrap_angle(float(yaw) + offset_yaw)

    def finish_standoff(self, goal_x, goal_y, yaw):
        """
        Drive the last stretch to the standoff when Nav2 stopped short.

        Only called after Nav2 reported the base near the goal and cancelled.
        The goal passed the map clearance check, /odom is ground truth and
        the remaining distance is bounded, so a slow straight drive is safer
        than letting the arm-to-cube distance decide the grasp.
        """
        if self.odom is None:
            self.get_logger().error('没有 /odom，无法补完最后一段导航')
            return False
        goal = self.map_pose_in_odom(goal_x, goal_y, yaw)
        if goal is None:
            self.get_logger().error(
                '没有 %s -> %s 的 TF，无法补完最后一段导航'
                % (ODOM_FRAME, MAP_FRAME))
            return False
        start = odom_xy(self.odom)
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if distance > FINISH_DISTANCE_LIMIT:
            self.get_logger().warning(
                '离导航目标还有 %.2f m，超过 %.2f m 的直接补完上限，'
                '就在当前位置抓取'
                % (distance, FINISH_DISTANCE_LIMIT))
            return False
        if distance > FINISH_TOLERANCE:
            # Nav2 may still be coasting on its last velocity command when the
            # goal reports SUCCEEDED (the base has no brake), and any leftover
            # command fights the in-place turn. Stop first so the rotation is
            # measured against a still base.
            self.stop_and_settle()
            start = odom_xy(self.odom)
            distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
            heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
            self.turn_by(wrap_angle(heading - odom_yaw(self.odom)),
                         timeout=FINISH_TURN_TIMEOUT)
            self.get_logger().info(
                'Nav2 停在目标 %.2f m 外，直接低速补完最后一段 ...' % distance)
            start = odom_xy(self.odom)
            travelled = 0.0
            deadline = time.monotonic() + FINISH_TIMEOUT
            while rclpy.ok() and time.monotonic() < deadline:
                if travelled >= distance - FINISH_TOLERANCE:
                    break
                self.cmd_pub.publish(make_forward_twist(FINISH_SPEED))
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.odom is not None:
                    travelled = odom_distance(start, odom_xy(self.odom))
            self.stop_and_settle()
            if travelled < distance - FINISH_TOLERANCE:
                self.get_logger().warning(
                    '只补了 %.2f m（目标 %.2f m），继续用视觉对准' %
                    (travelled, distance))
        if self.odom is not None:
            remaining = wrap_angle(goal[2] - odom_yaw(self.odom))
            if abs(remaining) > FINISH_YAW_TOLERANCE:
                self.turn_by(remaining, timeout=FINISH_TURN_TIMEOUT)
                self._publish_stop(0.3)
        return True

    def aim_at_estimated_cube(self, cube_xy):
        """
        Point the wrist camera at the cube from the stopped base.

        Prefers the search-time world point, which comes from ground-truth
        odometry. The map estimate carries AMCL's drift -- the very error
        that stopped Nav2 short -- so aiming by map alone can miss the cube
        by more than the camera's +/-8 cm view.
        """
        cube = None
        source = '地图估计'
        if self.cube_world is not None and self.base_pose() is not None:
            point = self.point_in_arm_frame(self.cube_world)
            cube = (point.x, point.y, point.z)
            source = '搜索时的世界坐标'
        if cube is None:
            transform = self.lookup_map_transform()
            if transform is None:
                self.get_logger().error(
                    '没有 %s -> %s 的 TF，无法把相机对准方块'
                    % (MAP_FRAME, ARM_BASE_FRAME))
                return False
            cube = transform_point(
                transform, (float(cube_xy[0]), float(cube_xy[1]), 0.0))
        look = fallback_look_position(cube, self.look_position[2])
        if look is None:
            self.get_logger().error(
                '底盘停在离目标较远处，方块在 %s 下为 (%.2f, %.2f)，'
                '超出可抓范围（x %.2f-%.2f m，|y| <= %.2f m），'
                '不再盲目抓取。请检查 Nav2 是否能到达 standoff。'
                % (ARM_BASE_FRAME, cube[0], cube[1], GRASP_REACH_MIN,
                   GRASP_REACH_MAX, GRASP_LATERAL_MAX))
            return False
        self.get_logger().warning(
            '方块当前在 %s 下为 (%.2f, %.2f)（来源：%s），把腕部相机'
            '观察点从 (%.2f, %.2f, %.2f) 改到 (%.2f, %.2f, %.2f) ...'
            % (ARM_BASE_FRAME, cube[0], cube[1], source,
               self.look_position[0], self.look_position[1],
               self.look_position[2], look[0], look[1], look[2]))
        self.look_position = look
        return True

    def navigate_to_cube(self, cube_xy):
        """Drive to the standoff point in front of the cube."""
        robot_xy = self.map_robot_xy()
        if robot_xy is None:
            self.get_logger().error(
                '没有 %s -> %s 的 TF，无法计算导航目标'
                % (MAP_FRAME, ROBOT_FRAME))
            return False
        goal_x, goal_y, yaw = standoff_goal(
            cube_xy, robot_xy, self.standoff)
        self.get_logger().info(
            '导航：方块在 (%0.2f, %0.2f)，机器人在 (%0.2f, %0.2f)，'
            '目标停在离方块 %.2f m 的 (%0.2f, %0.2f)，朝向 %.0f° ...'
            % (cube_xy[0], cube_xy[1], robot_xy[0], robot_xy[1],
               self.standoff, goal_x, goal_y, math.degrees(yaw)))
        if not self.wait_for_occupancy_map():
            self.get_logger().error(
                '没有收到 %s，无法确认目标在墙内，不发送导航目标'
                % MAP_TOPIC)
            return False
        free, reason = self.goal_is_free(goal_x, goal_y)
        if free:
            # The nominal goal clears the bare minimum but may still fall in
            # the local costmap's inscribed band, in which case the warning
            # below reports this strict reason. Without it the log would say
            # "不可用：" with nothing after it.
            strict_free, strict_reason = self.goal_is_free(
                goal_x, goal_y, NAV_GOAL_SAFE_CLEARANCE)
            if not strict_free:
                reason = strict_reason or (
                    '离桌子/障碍物不足 %.2f m' % NAV_GOAL_SAFE_CLEARANCE)

        def first_free(clearance):
            """Nominal goal first, then the bearings/retreats at this margin."""
            if self.goal_is_free(goal_x, goal_y, clearance)[0]:
                return 0.0, 0.0, goal_x, goal_y, yaw
            for offset, extra, candidate_x, candidate_y, candidate_yaw in \
                    standoff_candidates(cube_xy, robot_xy, self.standoff):
                if offset == 0.0 and extra == 0.0:
                    continue
                if self.goal_is_free(candidate_x, candidate_y, clearance)[0]:
                    return offset, extra, candidate_x, candidate_y, \
                        candidate_yaw
            return None

        # Two things push the nominal goal into the table's clearance band:
        # the oblique top-face measurement (up to ~10 cm too far east) and a
        # base left at a shallow angle beside the table. Try the neighbouring
        # bearings and retreats, preferring goals whose cell the local
        # costmap can actually reach (NAV_GOAL_SAFE_CLEARANCE) and only then
        # the physically-safe bare minimum, because a goal in the inscribed
        # band wedges DWB. Every candidate keeps the standoff radius (the
        # arm's reach) and is re-checked against the map, so a candidate
        # inside the table or a wall is still refused.
        chosen = (first_free(NAV_GOAL_SAFE_CLEARANCE)
                  or first_free(NAV_GOAL_CLEARANCE))
        if chosen is not None:
            offset, extra, candidate_x, candidate_y, candidate_yaw = chosen
            if (offset, extra) != (0.0, 0.0):
                change = []
                if extra:
                    change.append('沿视线后退 %.2f m' % extra)
                if abs(offset) > 1e-6:
                    change.append('绕方块转 %.0f°' % math.degrees(offset))
                self.get_logger().warning(
                    '导航目标 (%0.2f, %0.2f) 不可用：%s；%s，改用 '
                    '(%0.2f, %0.2f) ...'
                    % (goal_x, goal_y, reason, '并'.join(change) or '调整',
                       candidate_x, candidate_y))
                goal_x, goal_y, yaw = candidate_x, candidate_y, candidate_yaw
        free = chosen is not None
        if not free:
            self.get_logger().error(
                '导航目标 (%0.2f, %0.2f) 不可用：%s。已拒绝发送，'
                '避免底盘撞墙或开出房间。' % (goal_x, goal_y, reason))
            return False
        status = self.send_nav_goal(goal_x, goal_y, yaw)
        if status == 'failed':
            return False
        if status == 'near':
            # Nav2 stopped within a few centimetres of the goal but never
            # reported SUCCEEDED (usually a yaw the base cannot settle on).
            # Drive the remaining, bounded stretch so the arm gets the
            # standoff geometry it was tuned for, then re-aim the camera
            # using the cube estimate and let the proven visual grasp run.
            self.finish_standoff(goal_x, goal_y, yaw)
            if not self.aim_at_estimated_cube(cube_xy):
                return False
            return True
        # SUCCEEDED does not mean the base is on the standoff. The stateful
        # goal checker latches the first moment both tolerances are met, and
        # this base has no brake: at y=+0.6 m it latched at the 5 cm edge of
        # the xy tolerance and then drifted 5.7 cm backwards over the next
        # 20 s, finishing 9 cm short and 0.81 m from the cube -- outside the
        # arm's 0.70 m reach, so the grasp was refused after a "successful"
        # navigation. Drive the bounded remainder from /odom, exactly as the
        # 'near' path does.
        self.finish_standoff(goal_x, goal_y, yaw)
        self.stop_and_settle()
        # Nav2 may stop up to yaw_goal_tolerance (0.25 rad = 14 deg) off the
        # bearing to the cube. That is harmless for the arm, which re-aligns
        # the gripper to the measured base yaw, but not for the wrist camera:
        # it looks almost straight down from about 14 cm above the cube top,
        # so its 60 deg view covers only +/-8 cm and a 10 cm lateral offset
        # puts the cube out of frame. The fixed look pose then fails all 10 s
        # of the colour search (observed at y=+0.6 m). Re-aim at the cube
        # using the ground-truth search-time point and the base's actual
        # pose, exactly as the 'near' path does.
        if not self.aim_at_estimated_cube(cube_xy):
            return False
        return True

    # --------------------------------------------------------------- retry
    def retreat(self, distance=RETREAT_DISTANCE):
        """Back the base away by ``distance`` metres, measured on /odom."""
        if self.odom is None:
            self.get_logger().error('没有 /odom，无法后退')
            return False
        start_xy = odom_xy(self.odom)
        deadline = time.monotonic() + RETREAT_TIMEOUT
        travelled = 0.0
        self.get_logger().info('后退 %.0f cm，重新观察 ...'
                               % (distance * 100.0))
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_pub.publish(make_forward_twist(-RETREAT_SPEED))
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom is None:
                continue
            travelled = odom_distance(start_xy, odom_xy(self.odom))
            if travelled >= distance:
                break
        self.stop_and_settle()
        if travelled < distance - 0.02:
            self.get_logger().warning(
                '只后退了 %.1f cm（目标 %.0f cm）'
                % (travelled * 100.0, distance * 100.0))
        return True

    # ----------------------------------------------------------------- run
    def run(self):
        """Search, drive, grasp; retry the grasp once after backing off."""
        self.get_logger().info('等待 /odom ...')
        if not self.wait_for_odom():
            self.get_logger().error('超时：没有收到 /odom')
            return False
        self.get_logger().info('等待 Panda 关节状态 ...')
        if not self.wait_for_arm_joint_states():
            self.get_logger().error('超时：/joint_states 中没有 Panda 关节')
            return False
        self.get_logger().info('等待 MoveIt /move_action ...')
        if not self.wait_for_move_group():
            self.get_logger().error('超时：没有找到 /move_action')
            return False

        self.get_logger().info('第零步：机械臂回 Home 姿态 ...')
        if self.home_first and not self.send_home_trajectory():
            self.get_logger().error('回 Home 失败')
            return False
        if not self.send_move_goal(make_gripper_goal(GRIPPER_OPEN)):
            self.get_logger().error('张开夹爪失败')
            return False
        if not self.ensure_gripper_open('张开夹爪后'):
            self.get_logger().error('夹爪没有完全张开')
            return False

        if not self.wait_for_map():
            return False

        cube_xy = self.search_for_cube()
        if cube_xy is None:
            return False

        # Fold the arm before driving: the reach pose sticks 0.4 m ahead of
        # the base and Nav2 plans for a 0.22 m radius footprint.
        self.get_logger().info('第一步：收起机械臂，准备导航 ...')
        if not self.send_home_trajectory():
            self.get_logger().error('导航前回 Home 失败')
            return False

        self.get_logger().info('第二步：用 Nav2 导航到方块附近 ...')
        if not self.navigate_to_cube(cube_xy):
            return False

        for attempt in range(1, GRASP_ATTEMPTS + 1):
            self.get_logger().info(
                '第三步：视觉抓取（第 %d/%d 次）...'
                % (attempt, GRASP_ATTEMPTS))
            if self.grasp_after_arrival():
                self.get_logger().info(
                    '找-走-抓完成：方块已被夹爪提起。')
                return True
            if attempt == GRASP_ATTEMPTS:
                break
            self.get_logger().warning(
                '抓取失败，后退 %.0f cm 后重新导航到方块前再试一次 ...'
                % (RETREAT_DISTANCE * 100.0))
            # The failed grasp leaves the gripper down around the cube.
            # Backing the base off now would drag the pads through it and
            # knock it over, so the retry has to start by folding the arm.
            self.get_logger().info('后退前先收起机械臂 ...')
            if not self.send_home_trajectory():
                self.get_logger().error('后退前回 Home 失败')
                return False
            if not self.retreat():
                self.get_logger().error('后退失败，无法重试抓取')
                return False
            # Backing off leaves the robot ~0.7 m from the cube, where the
            # wrist camera sees only a small, edge-on view. Drive back to the
            # standoff so the retry starts from the geometry the grasp was
            # tuned for.
            if not self.navigate_to_cube(cube_xy):
                self.get_logger().error('后退后重新导航失败，无法重试抓取')
                return False
        self.get_logger().error('视觉抓取失败：重试后仍未夹起方块。')
        return False


def time_msg():
    """Zero time: tf2 reads it as "latest available"."""
    return Time()


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = SearchAndGraspNode()
    exit_code = 1
    try:
        exit_code = 0 if node.run() else 1
    except KeyboardInterrupt:
        node.get_logger().warning('用户中断')
    finally:
        try:
            node.stop_and_settle()
        except Exception:  # noqa: BLE001 - shutdown must never mask the result
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
