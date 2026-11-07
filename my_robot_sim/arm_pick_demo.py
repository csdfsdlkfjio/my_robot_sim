"""
用 move_group 控制 Panda 机械臂抓取场景里的一个虚拟箱子.

流程：创建箱子 → 张开夹爪 → 回 Home → 到箱子上方 → 下探 → 闭合夹爪 →
把箱子挂到 panda_hand 上 → 抬起 → 退出时卸载箱子。

注意：ROS 2 Humble **没有** moveit_commander 包（moveit_commander 是 ROS 1 的
API，MoveIt 2 官方仓库和 apt 源里都没有）。这里用 rclpy 直接调用 move_group
提供的 ``/move_action``，它正是 RViz 拖拽规划和 moveit_commander 底层使用的
同一个接口；下面的 RobotCommander / MoveGroupCommander / PlanningSceneInterface
按 moveit_commander 的习惯提供了最小可用的同名类。
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
)
from moveit_msgs.srv import ApplyPlanningScene
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

BASE_FRAME = 'panda_link0'
END_EFFECTOR_LINK = 'panda_hand'

# 场景里的虚拟箱子：边长 5 cm 的方块，中心在 (0.5, 0, 0.25)
BOX_ID = 'pick_box'
BOX_SIZE = 0.05
BOX_CENTER = (0.5, 0.0, 0.25)

# panda_hand 原点到指尖平面的距离：手指关节装在 z=0.0584，手指网格再伸出
# 0.0538（实测 finger.stl 包围盒）。夹爪竖直朝下时，指尖在连杆下方这么多。
TOOL_LENGTH = 0.1122

# 夹爪的 z 轴就是夹爪朝向（指向两指之间）。默认朝向是 z 轴朝上，绕 base 的
# x 轴转 180° 后 z 轴朝下，夹爪垂直向下，像正常抓取一样俯视目标。
GRIPPER_ORIENTATION = (1.0, 0.0, 0.0, 0.0)  # (x, y, z, w)

# SRDF 里夹爪的规划组叫 "hand"（panda_hand 是连杆名，panda_hand_controller
# 是控制器名）。move_group 已把该组映射到 GripperCommand 控制器。
GRIPPER_GROUP = 'hand'
GRIPPER_JOINT = 'panda_finger_joint1'
GRIPPER_OPEN = 0.04  # m，手指行程上限
GRASP_GRIPPER = BOX_SIZE / 2  # 手指合到箱子两侧（5 cm 箱子对应 0.025）
LIFT_HEIGHT = 0.1  # m，抓取后抬起的高度
PRE_GRASP_CLEARANCE = 0.05  # m，抓取点先停在箱心正上方 5 cm

# 箱子挂到手上后，允许它与这几个连杆“接触”，不参与碰撞检查
TOUCH_LINKS = ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')

MOVE_ACTION = '/move_action'
APPLY_SCENE_SERVICE = '/apply_planning_scene'
JOINT_TOLERANCE = 0.01  # rad
POSITION_TOLERANCE = 0.01  # m，目标点允许的半径
ORIENTATION_TOLERANCE = 0.01  # rad，约 0.6°，保证夹爪真的是竖直朝下


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
    构造只约束位置的 PositionConstraint.

    constraint_region 用一个半径为 ``radius`` 的小球表示。
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


def make_orientation_constraint(link_name, quaternion=GRIPPER_ORIENTATION,
                                frame_id=BASE_FRAME,
                                tolerance=ORIENTATION_TOLERANCE):
    """
    构造姿态约束，把夹爪固定在指定朝向上.

    ``quaternion`` 是 (x, y, z, w)，含义是该连杆相对 ``frame_id`` 的朝向。
    默认的 (1, 0, 0, 0) 让连杆 z 轴朝下，也就是夹爪垂直向下。
    """
    constraint = OrientationConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.orientation = Quaternion(
        x=float(quaternion[0]), y=float(quaternion[1]),
        z=float(quaternion[2]), w=float(quaternion[3]))
    constraint.absolute_x_axis_tolerance = tolerance
    constraint.absolute_y_axis_tolerance = tolerance
    constraint.absolute_z_axis_tolerance = tolerance
    constraint.weight = 1.0
    return constraint


def lift_position(position, height=LIFT_HEIGHT):
    """返回在 ``position`` 基础上垂直抬高 ``height`` 的位置."""
    return (position[0], position[1], position[2] + height)


def tcp_to_hand_position(tcp_position, tool_length=TOOL_LENGTH):
    """把两指中点（抓取点）的位置换算成 panda_hand 连杆的位置（夹爪朝下）."""
    return (tcp_position[0], tcp_position[1], tcp_position[2] + tool_length)


def grasp_hand_position(box_center=BOX_CENTER):
    """让指尖停到箱心时，panda_hand 连杆该在的位置."""
    return tcp_to_hand_position(box_center)


def pre_grasp_hand_position(box_center=BOX_CENTER,
                            clearance=PRE_GRASP_CLEARANCE):
    """抓取点停在箱心正上方 ``clearance`` 处时，panda_hand 连杆的位置."""
    return tcp_to_hand_position(
        (box_center[0], box_center[1], box_center[2] + clearance))


def make_box_object(object_id=BOX_ID, center=BOX_CENTER, size=BOX_SIZE,
                    frame_id=BASE_FRAME):
    """构造一个方块 CollisionObject（ADD 操作）."""
    pose = Pose()
    pose.position.x = float(center[0])
    pose.position.y = float(center[1])
    pose.position.z = float(center[2])
    pose.orientation.w = 1.0

    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    obj.primitives = [SolidPrimitive(
        type=SolidPrimitive.BOX, dimensions=[float(size)] * 3)]
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


def make_attached_object(object_id=BOX_ID, link_name=END_EFFECTOR_LINK,
                         operation=CollisionObject.ADD,
                         touch_links=TOUCH_LINKS, frame_id=BASE_FRAME):
    """
    构造挂载/卸载消息.

    ADD 表示把世界里的 ``object_id`` 挂到 ``link_name`` 上；REMOVE 表示卸下。
    挂载时如果物体已在世界里，MoveIt 会沿用世界里的位姿和几何，消息里不需要
    再写一遍。
    """
    attached = AttachedCollisionObject()
    attached.link_name = link_name
    attached.object.id = object_id
    attached.object.header.frame_id = frame_id
    attached.object.operation = operation
    attached.touch_links = list(touch_links)
    return attached


class PlanningSceneInterface:
    """最小版 PlanningSceneInterface：加箱子、挂载、卸载."""

    def __init__(self, node):
        self._node = node
        self._client = node.create_client(
            ApplyPlanningScene, APPLY_SCENE_SERVICE)

    def wait_for_server(self, timeout=10.0):
        """等待 move_group 的 /apply_planning_scene 可用."""
        return self._client.wait_for_service(timeout_sec=timeout)

    def _apply(self, scene):
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future)
        result = future.result()
        return bool(result and result.success)

    def add_box(self, object_id=BOX_ID, center=BOX_CENTER, size=BOX_SIZE):
        """把方块加入规划场景（作为障碍物）."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            make_box_object(object_id, center, size)]
        return self._apply(scene)

    def attach_box(self, object_id=BOX_ID, link_name=END_EFFECTOR_LINK):
        """把箱子挂到夹爪上，之后它会跟着手一起动."""
        return self._apply_attached(make_attached_object(
            object_id, link_name, CollisionObject.ADD))

    def detach_box(self, object_id=BOX_ID, link_name=END_EFFECTOR_LINK):
        """从夹爪上卸下箱子，箱子会回到世界里（停在卸下时的位置）."""
        return self._apply_attached(make_attached_object(
            object_id, link_name, CollisionObject.REMOVE))

    def _apply_attached(self, attached):
        scene = PlanningScene()
        scene.is_diff = True
        # 关键：MoveIt 2 的 setCurrentState() 会忽略
        # robot_state.is_diff == false 的所有非 ADD 操作，REMOVE 会被静默丢弃
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        return self._apply(scene)


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

    def go_to_joint_state(self, positions, joint_names=ARM_JOINTS):
        """规划并执行到一组关节角（等价于 moveit_commander 的 set_joint_value_target + go）."""
        constraints = Constraints(
            joint_constraints=make_joint_constraints(joint_names, positions))
        return self._move(constraints)

    def set_gripper(self, position):
        """闭合/张开夹爪：对 panda_finger_joint1 发目标，由 hand 组执行."""
        return self.go_to_joint_state([position], [GRIPPER_JOINT])

    def go_to_position(self, position, orientation=GRIPPER_ORIENTATION):
        """规划并执行到目标位置，并强制夹爪竖直朝下."""
        constraints = Constraints(
            position_constraints=[
                make_position_constraint(END_EFFECTOR_LINK, position)],
            orientation_constraints=[
                make_orientation_constraint(END_EFFECTOR_LINK, orientation)])
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
    """脚本入口：建箱子、回 Home、下探抓取、挂载、抬起、卸载."""
    rclpy.init(args=args)
    node = rclpy.create_node('arm_pick_demo')
    scene_interface = None
    attached = False
    try:
        robot = RobotCommander(node)
        group = MoveGroupCommander(node, 'panda_arm')
        hand = MoveGroupCommander(node, GRIPPER_GROUP)
        scene_interface = PlanningSceneInterface(node)

        node.get_logger().info('等待 move_group 的 %s ...' % MOVE_ACTION)
        if not group.wait_for_server(timeout=15.0):
            node.get_logger().error(
                '找不到 move_group 的 /move_action，'
                '请先在终端 1 启动 demo.launch.py')
            return 1
        if not scene_interface.wait_for_server(timeout=10.0):
            node.get_logger().error(
                '找不到 %s，move_group 没有提供规划场景服务'
                % APPLY_SCENE_SERVICE)
            return 1

        current = robot.get_current_joint_values()
        current_positions = [round(current[name], 3)
                             for name in ARM_JOINTS if name in current]
        node.get_logger().info('当前关节角: %s' % current_positions)

        node.get_logger().info(
            '创建 %.0f cm 的箱子，中心 (%.2f, %.2f, %.2f) ...'
            % (BOX_SIZE * 100, BOX_CENTER[0], BOX_CENTER[1], BOX_CENTER[2]))
        if not scene_interface.add_box():
            node.get_logger().error('把箱子加入规划场景失败')
            return 1

        # 夹爪上电时是闭合的，先张开，后面闭合时才能看出抓取动作
        node.get_logger().info('打开夹爪 ...')
        if not hand.set_gripper(GRIPPER_OPEN):
            node.get_logger().error('张开夹爪失败')
            return 1

        node.get_logger().info('第一步：回到 Home 姿态 ...')
        if not group.go_to_joint_state(HOME_JOINTS):
            node.get_logger().error('回 Home 失败')
            return 1

        pre_grasp = pre_grasp_hand_position()
        node.get_logger().info(
            '第二步：移动到箱子上方，抓取点 z=%.2f ...'
            % (pre_grasp[2] - TOOL_LENGTH))
        if not group.go_to_position(pre_grasp):
            node.get_logger().error('移动到箱子上方失败')
            return 1

        grasp = grasp_hand_position()
        node.get_logger().info(
            '第三步：下探到箱子中心，抓取点 z=%.2f ...'
            % (grasp[2] - TOOL_LENGTH))
        if not group.go_to_position(grasp):
            node.get_logger().error('下探失败')
            return 1

        node.get_logger().info('第四步：闭合夹爪 ...')
        if not hand.set_gripper(GRASP_GRIPPER):
            node.get_logger().error('闭合夹爪失败')
            return 1
        node.get_logger().info('夹爪已闭合')

        node.get_logger().info('第五步：把箱子挂到 %s 上 ...'
                               % END_EFFECTOR_LINK)
        if not scene_interface.attach_box():
            node.get_logger().error('挂载箱子失败')
            return 1
        attached = True
        node.get_logger().info('箱子已挂到夹爪上')

        lifted = lift_position(grasp)
        node.get_logger().info(
            '第六步：向上抬起 %.2f m 到 z=%.2f ...'
            % (LIFT_HEIGHT, lifted[2]))
        if not group.go_to_position(lifted):
            node.get_logger().error('抬起失败')
            return 1
        node.get_logger().info('抓取并抬起完成！')
        return 0
    finally:
        if attached and scene_interface is not None:
            node.get_logger().info('退出前卸载箱子 ...')
            if not scene_interface.detach_box():
                node.get_logger().error('卸载箱子失败')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
