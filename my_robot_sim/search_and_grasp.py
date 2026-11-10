"""
Search for the red cube, drive to it with Nav2, then grasp it.

Stages:
    1. search   - aim the wrist camera forward and turn the base in place a
                  full turn, stopping at the first red blob;
    2. navigate - send the cube's map position minus a standoff distance to
                  Nav2's NavigateToPose action and wait for the base;
    3. grasp    - run the visual-grasp sequence (VisualGraspNode) on the
                  settled base;
    4. retry    - back away 20 cm, drive back to the standoff and run the
                  grasp once more if it failed.

Why the camera and not the lidar: lidar_link sits 5 cm above base_link, which
is 0.20 m high, so the scan plane is at z ~= 0.25 m -- it grazes the cube
(0.225-0.275 m) and only a couple of rays can hit it. A scan can also never
identify the cube as red: the pedestal, the walls and the blue/orange boxes
all come back as obstacles. The wrist camera is the sensor that both sees the
small cube and knows its colour.

Usage:
    world=$(ros2 pkg prefix --share my_robot_sim)/worlds/demo.world
    ros2 launch my_robot_sim mobile_manipulation.launch.py world:=$world
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
from rclpy.action import ActionClient
from tf2_ros import TransformException

from my_robot_sim.mobile_manipulation_node import (
    ARM_BASE_FRAME,
    GRIPPER_OPEN,
    TF_TIMEOUT,
    make_forward_twist,
    make_gripper_goal,
    make_reach_goal,
    odom_distance,
    odom_xy,
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

# Nav2 goal distance from the cube centre. The Panda reaches about 0.85 m
# from panda_link0, and the visual grasp expects the cube roughly 0.45-0.55 m
# in front of base_link (its proven run had it 0.43 m ahead), so 1 m would be
# out of reach.
STANDOFF_DISTANCE = 0.5  # m
NAV_START_TIMEOUT = 30.0  # s for the goal to be accepted
NAV_TIMEOUT = 180.0  # s for the goal to finish
NAV_FEEDBACK_PERIOD = 3.0  # s between distance logs

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


def standoff_goal(cube_xy, robot_xy, standoff=STANDOFF_DISTANCE):
    """
    Return (x, y, yaw) for the Nav2 goal ``standoff`` metres from the cube.

    The goal sits on the line from the cube to the robot, so the base keeps
    the side it already sees the cube from, and it faces the cube.
    """
    dx = float(robot_xy[0]) - float(cube_xy[0])
    dy = float(robot_xy[1]) - float(cube_xy[1])
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        # Degenerate: the robot thinks it is on top of the cube. Approach
        # from -x, which matches the direction it was driving.
        ux, uy = -1.0, 0.0
    else:
        ux, uy = dx / distance, dy / distance
    goal_x = float(cube_xy[0]) + float(standoff) * ux
    goal_y = float(cube_xy[1]) + float(standoff) * uy
    yaw = math.atan2(float(cube_xy[1]) - goal_y,
                     float(cube_xy[0]) - goal_x)
    return goal_x, goal_y, yaw


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
        self.map_transform = None

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
        """Log the remaining distance a few times per navigation."""
        now = time.monotonic()
        if now - self.last_nav_feedback < NAV_FEEDBACK_PERIOD:
            return
        self.last_nav_feedback = now
        self.get_logger().info(
            '导航中：剩余 %.2f m'
            % float(feedback.feedback.distance_remaining))

    def send_nav_goal(self, x, y, yaw, timeout=NAV_TIMEOUT):
        """Send one NavigateToPose goal and return True when it succeeds."""
        if not self.nav_client.wait_for_server(
                timeout_sec=NAV_START_TIMEOUT):
            self.get_logger().error(
                '找不到 Nav2 的 %s Action。先启动 '
                'ros2 launch my_robot_sim nav2.launch.py' % NAV_ACTION)
            return False
        goal = make_navigate_goal(x, y, yaw)
        send_future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._on_nav_feedback)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=NAV_START_TIMEOUT)
        if not send_future.done():
            self.get_logger().error('发送导航目标超时')
            return False
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Nav2 拒绝了导航目标')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error(
                '导航 %.0f s 未完成，取消目标' % timeout)
            handle.cancel_goal_async()
            self.stop_and_settle()
            return False
        status = result_future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('导航失败，Action 状态 %d' % status)
            return False
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
        if not self.send_nav_goal(goal_x, goal_y, yaw):
            return False
        self.stop_and_settle()
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
