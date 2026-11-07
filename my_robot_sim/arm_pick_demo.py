"""
用 move_group 控制 Panda 机械臂：先回 Home 姿态，再到目标点.

注意：ROS 2 Humble **没有** moveit_commander 包（moveit_commander 是 ROS 1 的
API，MoveIt 2 官方仓库和 apt 源里都没有）。这里用 rclpy 直接调用 move_group
提供的 ``/move_action``，它正是 RViz 拖拽规划和 moveit_commander 底层使用的
同一个接口；下面的 RobotCommander / MoveGroupCommander 按 moveit_commander
的习惯提供了最小可用的同名类。
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    PositionConstraint,
)
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

# Panda 官方 demo 的初始关节角（moveit_resources_panda_moveit_config/
# config/initial_positions.yaml），也就是 Home 姿态
HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

ARM_JOINTS = (
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
)

TARGET_POSITION = (0.5, 0.0, 0.3)
BASE_FRAME = 'panda_link0'
END_EFFECTOR_LINK = 'panda_hand'

MOVE_ACTION = '/move_action'
JOINT_TOLERANCE = 0.01  # rad
POSITION_TOLERANCE = 0.01  # m，目标点允许的半径


def make_joint_constraints(joint_names, positions, tolerance=JOINT_TOLERANCE):
    """构造一组关节目标约束（对应 moveit_commander 的关节目标）."""
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
                             frame_id=BASE_FRAME,
                             radius=POSITION_TOLERANCE):
    """
    构造只约束位置、不约束姿态的 PositionConstraint.

    constraint_region 用一个半径为 ``radius`` 的小球表示；姿态完全放开，
    这样最容易规划到目标点。
    """
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


class RobotCommander:
    """最小版 RobotCommander：读取当前关节角（替代 moveit_commander 的类）."""

    def __init__(self, node):
        self._node = node
        self._latest = None
        self._subscription = node.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

    def _on_joint_state(self, msg):
        self._latest = msg

    def get_current_joint_values(self, timeout=10.0):
        """返回 {关节名: 位置}；收不到 /joint_states 时抛异常."""
        deadline = time.monotonic() + timeout
        while self._latest is None and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        if self._latest is None:
            raise RuntimeError('没有收到 /joint_states，move_group 或控制器没起来？')
        return dict(zip(self._latest.name, self._latest.position))


class MoveGroupCommander:
    """最小版 MoveGroupCommander：通过 /move_action 规划并执行."""

    def __init__(self, node, group_name):
        self._node = node
        self._group_name = group_name
        self._client = ActionClient(node, MoveGroup, MOVE_ACTION)

    def wait_for_server(self, timeout=10.0):
        """等待 move_group 的 /move_action 可用."""
        return self._client.wait_for_server(timeout_sec=timeout)

    def go_to_joint_state(self, positions):
        """规划并执行到一组关节角（等价于 moveit_commander 的 set_joint_value_target + go）."""
        constraints = Constraints(
            joint_constraints=make_joint_constraints(ARM_JOINTS, positions))
        return self._move(constraints)

    def go_to_position(self, position):
        """规划并执行到目标位置，姿态不限制."""
        constraints = Constraints(
            position_constraints=[
                make_position_constraint(END_EFFECTOR_LINK, position)])
        return self._move(constraints)

    def _move(self, constraints):
        goal = MoveGroup.Goal()
        goal.request.group_name = self._group_name
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        goal.request.goal_constraints = [constraints]
        # plan_only=False：让 move_group 规划完直接下发给控制器执行
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self._node.get_logger().error('move_group 拒绝了目标')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)
        result = result_future.result()
        error_code = result.result.error_code.val if result else None
        if error_code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(
                '规划或执行失败，MoveItErrorCodes=%s' % error_code)
            return False
        return True


def main(args=None):
    """脚本入口：回 Home，再去 (0.5, 0.0, 0.3)."""
    rclpy.init(args=args)
    node = rclpy.create_node('arm_pick_demo')
    try:
        robot = RobotCommander(node)
        group = MoveGroupCommander(node, 'panda_arm')

        node.get_logger().info('等待 move_group 的 %s ...' % MOVE_ACTION)
        if not group.wait_for_server(timeout=15.0):
            node.get_logger().error(
                '找不到 move_group 的 /move_action，'
                '请先在终端 1 启动 demo.launch.py')
            return 1

        current = robot.get_current_joint_values()
        current_positions = [round(current[name], 3)
                             for name in ARM_JOINTS if name in current]
        node.get_logger().info('当前关节角: %s' % current_positions)

        node.get_logger().info('第一步：回到 Home 姿态 ...')
        if not group.go_to_joint_state(HOME_JOINTS):
            node.get_logger().error('回 Home 失败')
            return 1

        node.get_logger().info(
            '第二步：前往目标点 x=%.2f, y=%.2f, z=%.2f ...' % TARGET_POSITION)
        if not group.go_to_position(TARGET_POSITION):
            node.get_logger().error('前往目标点失败')
            return 1

        node.get_logger().info('机械臂已到达目标点！')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
