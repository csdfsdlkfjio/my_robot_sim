"""机械臂演示脚本目标约束构造的单元测试."""

import math

from my_robot_sim.arm_pick_demo import (
    ARM_JOINTS, BASE_FRAME, END_EFFECTOR_LINK, GRIPPER_CLOSED, GRIPPER_JOINT,
    GRIPPER_OPEN, GRIPPER_ORIENTATION, HOME_JOINTS, JOINT_TOLERANCE,
    LIFT_HEIGHT, ORIENTATION_TOLERANCE, TARGET_POSITION, lift_position,
    make_joint_constraints, make_orientation_constraint,
    make_position_constraint)


def test_home_joints_match_panda_initial_positions():
    """Home 姿态必须是 Panda 官方 initial_positions.yaml 里的 7 个关节角."""
    assert HOME_JOINTS == (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    assert len(HOME_JOINTS) == len(ARM_JOINTS) == 7


def test_target_position_is_the_requested_point():
    """目标点必须是 x=0.5, y=0.0, z=0.3."""
    assert TARGET_POSITION == (0.5, 0.0, 0.3)


def test_arm_joint_names_are_panda_joints_1_to_7():
    """关节名必须按 panda_joint1..7 顺序排列."""
    assert ARM_JOINTS == tuple('panda_joint%d' % i for i in range(1, 8))


def test_joint_constraints_cover_every_joint():
    """每个关节都要有一条容差与权重正确的约束."""
    constraints = make_joint_constraints(ARM_JOINTS, HOME_JOINTS)
    assert [c.joint_name for c in constraints] == list(ARM_JOINTS)
    for constraint, expected in zip(constraints, HOME_JOINTS):
        assert constraint.position == expected
        assert constraint.tolerance_above == JOINT_TOLERANCE
        assert constraint.tolerance_below == JOINT_TOLERANCE
        assert constraint.weight == 1.0


def test_position_constraint_uses_sphere_at_target():
    """位置约束应是落在目标点上的小球，且绑定末端连杆."""
    constraint = make_position_constraint(END_EFFECTOR_LINK, TARGET_POSITION)
    assert constraint.header.frame_id == BASE_FRAME == 'panda_link0'
    assert constraint.link_name == END_EFFECTOR_LINK == 'panda_hand'

    region = constraint.constraint_region
    assert len(region.primitives) == len(region.primitive_poses) == 1
    assert region.primitives[0].type == region.primitives[0].SPHERE

    center = region.primitive_poses[0].position
    assert (center.x, center.y, center.z) == TARGET_POSITION


def test_gripper_orientation_is_unit_quaternion():
    """夹爪朝向的四元数必须是单位长度，否则约束无意义."""
    x, y, z, w = GRIPPER_ORIENTATION
    assert abs(math.sqrt(x * x + y * y + z * z + w * w) - 1.0) < 1e-9


def test_gripper_orientation_points_straight_down():
    """绕 x 轴 180°：连杆 z 轴（夹爪朝向）必须指向 base 的 -z."""
    x, y, z, w = GRIPPER_ORIENTATION
    # 单位四元数 (x, y, z, w) 对应的旋转矩阵第三列就是 z 轴的像
    local_z_in_base = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    assert abs(local_z_in_base[0]) < 1e-9
    assert abs(local_z_in_base[1]) < 1e-9
    assert abs(local_z_in_base[2] + 1.0) < 1e-9


def test_orientation_constraint_binds_hand_in_base_frame():
    """姿态约束必须作用在末端连杆、base 坐标系上，且三轴容差一致."""
    constraint = make_orientation_constraint(END_EFFECTOR_LINK)
    assert constraint.header.frame_id == BASE_FRAME == 'panda_link0'
    assert constraint.link_name == END_EFFECTOR_LINK == 'panda_hand'

    q = constraint.orientation
    assert (q.x, q.y, q.z, q.w) == GRIPPER_ORIENTATION
    assert constraint.absolute_x_axis_tolerance == ORIENTATION_TOLERANCE
    assert constraint.absolute_y_axis_tolerance == ORIENTATION_TOLERANCE
    assert constraint.absolute_z_axis_tolerance == ORIENTATION_TOLERANCE
    assert constraint.weight == 1.0


def test_gripper_open_and_closed_stay_within_finger_limits():
    """闭合目标是 0，张开目标是手指行程上限 0.04 m."""
    assert GRIPPER_CLOSED == 0.0
    assert GRIPPER_OPEN == 0.04
    assert 0.0 <= GRIPPER_CLOSED <= GRIPPER_OPEN


def test_gripper_constraint_targets_finger_joint():
    """夹爪指令必须落在 panda_finger_joint1 上（hand 组实际驱动的关节）."""
    constraints = make_joint_constraints([GRIPPER_JOINT], [GRIPPER_CLOSED])
    assert len(constraints) == 1
    assert constraints[0].joint_name == 'panda_finger_joint1'
    assert constraints[0].position == 0.0


def test_lift_position_raises_only_z():
    """抓取后只在 z 轴抬高 0.1 m，x/y 不变."""
    lifted = lift_position(TARGET_POSITION)
    assert lifted == (TARGET_POSITION[0], TARGET_POSITION[1],
                      TARGET_POSITION[2] + LIFT_HEIGHT)
    assert lifted[2] == 0.4
