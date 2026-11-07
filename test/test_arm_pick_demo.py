"""机械臂演示脚本目标约束构造的单元测试."""

from my_robot_sim.arm_pick_demo import (
    ARM_JOINTS, BASE_FRAME, END_EFFECTOR_LINK, HOME_JOINTS, JOINT_TOLERANCE,
    TARGET_POSITION, make_joint_constraints, make_position_constraint)


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
