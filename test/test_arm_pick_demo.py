"""机械臂演示脚本目标位姿的单元测试."""

import math

from my_robot_sim.arm_pick_demo import (
    GOAL_FRAME, GOAL_POSITION, add_pipeline_params, execute_with_retry,
    make_goal_pose)


def test_goal_pose_position_and_frame():
    """目标位姿的坐标系与坐标必须与需求一致."""
    goal = make_goal_pose()
    assert goal.header.frame_id == GOAL_FRAME == 'panda_link0'
    assert (goal.pose.position.x, goal.pose.position.y,
            goal.pose.position.z) == GOAL_POSITION


def test_goal_orientation_is_unit_quaternion():
    """四元数必须是单位长度."""
    q = make_goal_pose().pose.orientation
    norm = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2)
    assert abs(norm - 1.0) < 1e-9


def test_goal_orientation_points_down():
    """绕 x 轴转 180°，末端法兰竖直朝下."""
    q = make_goal_pose().pose.orientation
    assert abs(q.y) < 1e-9 and abs(q.z) < 1e-9 and abs(q.w) < 1e-9
    assert abs(abs(q.x) - 1.0) < 1e-9


def test_pipeline_params_use_moveit_25_names():
    """流水线参数要用 MoveIt 2.5.10 认的名字."""
    config = add_pipeline_params({'planning_pipelines': ['ompl']})
    assert config['planning_pipelines.pipeline_names'] == ['ompl']
    assert config['plan_request_params.planning_pipeline'] == 'ompl'


def test_pipeline_params_do_not_mutate_input():
    """补参数时应返回新字典，不动调用方传入的数据."""
    original = {'planning_pipelines': ['ompl']}
    add_pipeline_params(original)
    assert original == {'planning_pipelines': ['ompl']}


def test_execute_with_retry_retries_until_success():
    """控制器还没连上时应该重试，直到 execute 成功."""
    calls = []

    def execute():
        calls.append(1)
        return len(calls) >= 3

    times = iter([0.0, 0.1, 0.2, 0.3])
    assert execute_with_retry(
        execute, timeout=10.0, interval=0.1,
        clock=lambda: next(times), sleep=lambda _: None)
    assert len(calls) == 3


def test_execute_with_retry_gives_up_after_timeout():
    """超过超时时间就返回失败，且只提示一次."""
    warnings = []
    times = iter([0.0, 5.0, 10.0])
    assert not execute_with_retry(
        lambda: False, timeout=10.0,
        clock=lambda: next(times), sleep=lambda _: None,
        on_retry=lambda: warnings.append(1))
    assert warnings == [1]
