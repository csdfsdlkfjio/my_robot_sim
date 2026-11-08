"""
Mobile-manipulation state machine.

Sequence:
    1. wait for /odom, /joint_states and MoveIt's /move_action;
    2. send the Panda home pose straight to its JointTrajectoryController
       (the Gazebo spawn pose is slightly outside the Panda joint limits and
       self-colliding, which MoveIt would reject as a start state);
    3. open both gripper fingers with MoveIt;
    4. drive forward ``drive_distance`` metres, checked against /odom;
    5. publish zero velocity and let the base settle;
    6. ask MoveIt to move panda_hand to ``target`` in the panda_link0 frame,
       with the gripper pointing straight down;
    7. close both gripper fingers.

Arm motion is executed by the two gazebo_ros2_control
JointTrajectoryControllers (homing directly, reaching through MoveIt). The
base is still driven by the Gazebo differential-drive plugin, so the two
control paths never touch the same joint.
"""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, Quaternion, Twist
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = tuple('panda_joint%d' % index for index in range(1, 8))
HAND_JOINTS = ('panda_finger_joint1', 'panda_finger_joint2')

# Same home pose as the Panda MoveIt configuration.
HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

ARM_GROUP = 'panda_arm'
HAND_GROUP = 'hand'
ARM_BASE_FRAME = 'panda_link0'
END_EFFECTOR_LINK = 'panda_hand'

# The homing motion bypasses MoveIt and talks to the ros2_control JTC directly.
ARM_TRAJECTORY_ACTION = '/panda_arm_controller/follow_joint_trajectory'
HOME_DURATION = 4.0  # s
HOME_TIMEOUT = 30.0  # s

# Target expressed in the arm base frame (panda_link0), i.e. 0.5 m in front of
# the arm and 0.3 m up. Orientation (x=1, w=0) points the gripper z axis down.
TARGET_POSITION = (0.5, 0.0, 0.3)
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)

# Panda finger travel is 0..0.04 m (0 = fingers together). The node opens the
# gripper before driving and closes it to 5 mm per finger at the target.
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.005

DRIVE_DISTANCE = 1.0
FORWARD_SPEED = 0.25
STOP_PUBLISH_SECONDS = 1.0
SETTLE_SECONDS = 0.5
DRIVE_TIMEOUT = 30.0
MOVE_TIMEOUT = 90.0

POSITION_TOLERANCE = 0.01  # m
ORIENTATION_TOLERANCE = 0.05  # rad, about 2.9 degrees
JOINT_TOLERANCE = 0.01  # rad or m


def odom_distance(start_xy, current_xy):
    """Return the planar distance between two (x, y) points."""
    return math.hypot(current_xy[0] - start_xy[0],
                      current_xy[1] - start_xy[1])


def make_joint_constraints(joint_names, positions, tolerance=JOINT_TOLERANCE):
    """Build one JointConstraint per joint."""
    return [
        JointConstraint(
            joint_name=name,
            position=float(position),
            tolerance_above=tolerance,
            tolerance_below=tolerance,
            weight=1.0,
        )
        for name, position in zip(joint_names, positions)
    ]


def make_position_constraint(link_name, position,
                             frame_id=ARM_BASE_FRAME,
                             radius=POSITION_TOLERANCE):
    """Constrain ``link_name`` to a small sphere around ``position``."""
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[radius])
    sphere_pose = Pose()
    sphere_pose.position.x = float(position[0])
    sphere_pose.position.y = float(position[1])
    sphere_pose.position.z = float(position[2])
    sphere_pose.orientation.w = 1.0

    constraint = PositionConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.constraint_region = BoundingVolume(
        primitives=[sphere], primitive_poses=[sphere_pose])
    constraint.weight = 1.0
    return constraint


def make_orientation_constraint(link_name, quaternion=GRIPPER_DOWN,
                                frame_id=ARM_BASE_FRAME,
                                tolerance=ORIENTATION_TOLERANCE):
    """Constrain ``link_name`` to a fixed orientation."""
    x, y, z, w = quaternion
    constraint = OrientationConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.orientation = Quaternion(
        x=float(x), y=float(y), z=float(z), w=float(w))
    constraint.absolute_x_axis_tolerance = tolerance
    constraint.absolute_y_axis_tolerance = tolerance
    constraint.absolute_z_axis_tolerance = tolerance
    constraint.weight = 1.0
    return constraint


def make_move_goal(group_name, constraints, attempts=10,
                   planning_time=10.0, velocity=0.3, acceleration=0.3):
    """Wrap constraints into a MoveGroup goal that also executes the plan."""
    goal = MoveGroup.Goal()
    goal.request.group_name = group_name
    goal.request.num_planning_attempts = attempts
    goal.request.allowed_planning_time = planning_time
    goal.request.max_velocity_scaling_factor = velocity
    goal.request.max_acceleration_scaling_factor = acceleration
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = False
    goal.planning_options.replan = False
    return goal


def make_home_trajectory(duration=HOME_DURATION):
    """Build a JTC goal that drives the arm to the Panda home pose."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [float(position) for position in HOME_JOINTS]
    point.time_from_start = Duration(seconds=duration).to_msg()
    goal.trajectory.points = [point]
    return goal


def make_reach_goal(position=TARGET_POSITION):
    """Move the hand to ``position`` with the gripper pointing down."""
    constraints = Constraints(
        position_constraints=[
            make_position_constraint(END_EFFECTOR_LINK, position)],
        orientation_constraints=[
            make_orientation_constraint(END_EFFECTOR_LINK)],
    )
    return make_move_goal(ARM_GROUP, constraints, attempts=10,
                          planning_time=10.0)


def make_gripper_goal(position=GRIPPER_CLOSED):
    """Close (or open) both fingers symmetrically."""
    constraints = Constraints(
        joint_constraints=make_joint_constraints(
            HAND_JOINTS, [position, position], tolerance=0.005))
    return make_move_goal(HAND_GROUP, constraints, attempts=1,
                          planning_time=2.0, velocity=0.5, acceleration=0.5)


def make_forward_twist(speed=FORWARD_SPEED):
    """Constant forward velocity command."""
    command = Twist()
    command.linear.x = float(speed)
    command.angular.z = 0.0
    return command


def make_stop_twist():
    """All-zero velocity command."""
    return Twist()


def has_all_joints(joint_state, joint_names):
    """Check that every name in ``joint_names`` is present in the state."""
    if joint_state is None:
        return False
    available = set(joint_state.name)
    return all(name in available for name in joint_names)


def odom_xy(odometry):
    """Extract (x, y) from an Odometry message."""
    position = odometry.pose.pose.position
    return (position.x, position.y)


class MobileManipulationNode(Node):
    """Run the drive -> stop -> reach -> grasp sequence."""

    def __init__(self):
        super().__init__('mobile_manipulation_node')
        self.drive_distance = float(
            self.declare_parameter('drive_distance', DRIVE_DISTANCE).value)
        self.forward_speed = float(
            self.declare_parameter('forward_speed', FORWARD_SPEED).value)
        target_x = float(
            self.declare_parameter('target_x', TARGET_POSITION[0]).value)
        target_y = float(
            self.declare_parameter('target_y', TARGET_POSITION[1]).value)
        target_z = float(
            self.declare_parameter('target_z', TARGET_POSITION[2]).value)
        self.target_position = (target_x, target_y, target_z)
        self.home_first = bool(
            self.declare_parameter('home_first', True).value)

        self.odom = None
        self.joint_state = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, ARM_TRAJECTORY_ACTION)

    def _on_odom(self, message):
        self.odom = message

    def _on_joint_state(self, message):
        if has_all_joints(message, ARM_JOINTS):
            self.joint_state = message

    def _spin_for(self, seconds):
        """Spin the node for a wall-clock duration."""
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_stop(self, seconds=STOP_PUBLISH_SECONDS):
        """Publish zero velocity repeatedly so the plugin really stops."""
        deadline = time.monotonic() + seconds
        stop = make_stop_twist()
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_odom(self, timeout=30.0):
        """Wait until at least one odometry message has arrived."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.odom is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.odom is not None

    def wait_for_arm_joint_states(self, timeout=30.0):
        """Wait for the first complete Panda /joint_states message."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.joint_state is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.joint_state is not None

    def wait_for_move_group(self, timeout=60.0):
        """Wait for MoveIt's /move_action server."""
        return self.move_client.wait_for_server(timeout_sec=timeout)

    def send_home_trajectory(self, timeout=HOME_TIMEOUT):
        """Home the arm through the JTC, bypassing MoveIt collision checks."""
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                '没有找到 %s' % ARM_TRAJECTORY_ACTION)
            return False

        send_future = self.arm_client.send_goal_async(make_home_trajectory())
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送 Home 轨迹超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('机械臂控制器拒绝了 Home 轨迹')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=HOME_DURATION + timeout)
        if not result_future.done():
            self.get_logger().error('等待 Home 轨迹执行结果超时')
            handle.cancel_goal_async()
            return False

        status = result_future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Home 轨迹执行失败，status=%d' % status)
            return False
        return True

    def send_move_goal(self, goal, timeout=MOVE_TIMEOUT):
        """Send a MoveGroup goal and wait for execution to finish."""
        send_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送 MoveGroup 目标超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveIt 拒绝了目标')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error('等待 MoveIt 执行结果超时')
            handle.cancel_goal_async()
            return False

        result = result_future.result()
        error_code = result.result.error_code.val if result else None
        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                'MoveIt 规划/执行失败，MoveItErrorCodes=%s' % error_code)
            return False
        return True

    def drive_forward(self, distance=None):
        """Drive straight until /odom reports ``distance`` metres travelled."""
        if distance is None:
            distance = self.drive_distance
        if self.odom is None:
            self.get_logger().error('没有 /odom，无法开始行驶')
            return False

        start_xy = odom_xy(self.odom)
        deadline = time.monotonic() + DRIVE_TIMEOUT
        self.get_logger().info(
            '开始前进 %.2f m（速度 %.2f m/s）...'
            % (distance, self.forward_speed))
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_pub.publish(make_forward_twist(self.forward_speed))
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom is None:
                continue
            travelled = odom_distance(start_xy, odom_xy(self.odom))
            if travelled >= distance:
                self.get_logger().info(
                    '已行驶 %.3f m，停车。' % travelled)
                return True

        self.get_logger().error(
            '行驶 %.1f s 仍未到达 %.2f m，停车。' % (DRIVE_TIMEOUT, distance))
        return False

    def stop_and_settle(self):
        """Send zero velocity, then wait for the base to stop moving."""
        self.get_logger().info('发送零速度并等待底盘停稳 ...')
        self._publish_stop(STOP_PUBLISH_SECONDS)
        self._spin_for(SETTLE_SECONDS)

    def run(self):
        """Execute the complete mobile-manipulation sequence."""
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

        if self.home_first:
            self.get_logger().info('第一步：机械臂回 Home 姿态 ...')
            if not self.send_home_trajectory():
                self.get_logger().error('回 Home 失败')
                return False

        self.get_logger().info('第二步：张开夹爪 ...')
        if not self.send_move_goal(make_gripper_goal(GRIPPER_OPEN)):
            self.get_logger().error('张开夹爪失败')
            return False

        self.get_logger().info('第三步：向前行驶 ...')
        if not self.drive_forward():
            self.stop_and_settle()
            return False

        self.stop_and_settle()

        self.get_logger().info(
            '第四步：手臂伸向 (%0.2f, %0.2f, %0.2f) [panda_link0] ...'
            % self.target_position)
        if not self.send_move_goal(make_reach_goal(self.target_position)):
            self.get_logger().error('伸向目标点失败')
            return False

        self.get_logger().info('第五步：闭合夹爪 ...')
        if not self.send_move_goal(make_gripper_goal()):
            self.get_logger().error('闭合夹爪失败')
            return False

        self.get_logger().info('移动抓取流程完成。')
        return True


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = MobileManipulationNode()
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
    raise SystemExit(main())
