"""用 MoveItPy 让 Panda 机械臂移动到指定位姿的演示."""

import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

GOAL_FRAME = 'panda_link0'
GOAL_POSITION = (0.5, 0.0, 0.3)
# 绕 x 轴转 180°，让 panda_hand 的 z 轴竖直向下（夹爪朝下）
GOAL_ORIENTATION = (1.0, 0.0, 0.0, 0.0)

DEFAULT_PIPELINE = 'ompl'
JOINT_STATES_TIMEOUT = 10.0
EXECUTE_RETRY_TIMEOUT = 15.0
EXECUTE_RETRY_INTERVAL = 0.25


def make_goal_pose(frame_id=GOAL_FRAME, position=GOAL_POSITION,
                   orientation=GOAL_ORIENTATION):
    """按给定参数构造目标位姿消息."""
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.pose.position.x = position[0]
    goal.pose.position.y = position[1]
    goal.pose.position.z = position[2]
    goal.pose.orientation.x = orientation[0]
    goal.pose.orientation.y = orientation[1]
    goal.pose.orientation.z = orientation[2]
    goal.pose.orientation.w = orientation[3]
    return goal


def add_pipeline_params(config_dict, pipeline=DEFAULT_PIPELINE):
    """
    补上 MoveItPy 2.5.10 需要的规划流水线参数.

    moveit_configs_utils 生成的是 ``planning_pipelines: [ompl]``，但
    MoveIt 2.5 的 MoveItCpp 读的是 ``planning_pipelines.pipeline_names``，
    PlanningComponent 读的是 ``plan_request_params.planning_pipeline``。
    缺少这两个参数时会分别报：

    - Failed to load any planning pipelines
    - No planning pipeline available for name ''

    返回新字典，不修改传入的 ``config_dict``。
    """
    config = dict(config_dict)
    config['planning_pipelines.pipeline_names'] = [pipeline]
    config['plan_request_params.planning_pipeline'] = pipeline
    return config


def execute_with_retry(execute, timeout=EXECUTE_RETRY_TIMEOUT,
                       interval=EXECUTE_RETRY_INTERVAL,
                       clock=time.monotonic, sleep=time.sleep,
                       on_retry=None):
    """
    反复调用 execute，直到成功或超时.

    MoveItPy 启动后，它内部的控制器 action client 需要一点时间才能完成
    DDS 发现；这期间 execute 会返回 ABORTED 并打印
    "Action client not connected to action server"。此时轨迹并没有被执行，
    所以可以安全地重试。``on_retry`` 只会被调用一次，避免刷屏。
    """
    deadline = clock() + timeout
    while True:
        if execute():
            return True
        if clock() >= deadline:
            return False
        if on_retry is not None:
            on_retry()
            on_retry = None
        sleep(interval)


class PickDemo(Node):
    """MoveItPy 客户端节点."""

    def __init__(self):
        super().__init__('pick_demo')
        # MoveItPy 是独立进程，不会去读 move_group 的参数服务器。
        # robot_description / robot_description_semantic / 运动学参数
        # 必须自己通过 config_dict 传进去，否则会报
        # "Could not find parameter robot_description_semantic"
        # 以及 "Unable to configure planning scene monitor"。
        from moveit.planning import MoveItPy
        from moveit_configs_utils import MoveItConfigsBuilder

        moveit_config = (
            MoveItConfigsBuilder('moveit_resources_panda')
            .planning_pipelines(pipelines=[DEFAULT_PIPELINE])
            .to_moveit_configs()
        )
        self.moveit = MoveItPy(
            node_name='pick_demo_moveit',
            config_dict=add_pipeline_params(moveit_config.to_dict()),
        )
        self.arm = self.moveit.get_planning_component('panda_arm')
        self.get_logger().info('MoveIt 初始化完成！')

    def wait_for_joint_states(self, timeout=JOINT_STATES_TIMEOUT):
        """等第一帧 /joint_states，避免用未初始化的状态去规划."""
        received = []
        subscription = self.create_subscription(
            JointState, '/joint_states', received.append, 10)
        deadline = time.monotonic() + timeout
        try:
            while not received and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(subscription)
        if not received:
            self.get_logger().error('没有收到 /joint_states，无法规划')
            return False
        return True

    def _execute(self, trajectory):
        """执行轨迹，必要时等待控制器 action client 完成发现."""
        return execute_with_retry(
            lambda: bool(self.moveit.execute('panda_arm', trajectory)),
            on_retry=lambda: self.get_logger().warn(
                '控制器动作服务器尚未就绪，等待发现后重试...'),
        )

    def run(self):
        """规划并执行到目标位姿."""
        if not self.wait_for_joint_states():
            return False

        goal = make_goal_pose()
        goal.header.stamp = self.get_clock().now().to_msg()

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(
            pose_stamped_msg=goal, pose_link='panda_hand')

        self.get_logger().info(
            '正在规划到 x=%.2f, y=%.2f, z=%.2f ...' % GOAL_POSITION)
        plan_result = self.arm.plan()
        if not plan_result:
            self.get_logger().error('规划失败：目标点不可达或存在碰撞')
            return False

        self.get_logger().info('规划成功，正在执行...')
        if not self._execute(plan_result.trajectory):
            self.get_logger().error('执行失败：控制器未在超时时间内就绪')
            return False
        self.get_logger().info('机械臂已到达目标点！')
        return True


def main(args=None):
    """脚本入口."""
    rclpy.init(args=args)
    node = PickDemo()
    try:
        success = node.run()
    finally:
        node.moveit.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    sys.stdout.flush()
    # MoveIt 2.5.10 在解释器退出阶段销毁 pybind 对象时会触发
    # class_loader 的 abort（进程以 SIGABRT 结束）。跳过后面的
    # 解释器清理，脚本才能以正常退出码结束。
    os._exit(0 if success else 1)


if __name__ == '__main__':
    main()
