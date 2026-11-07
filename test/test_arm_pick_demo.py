"""机械臂抓取演示脚本的约束与抓取几何单元测试."""

import math

from my_robot_sim.arm_pick_demo import (
    ARM_JOINTS, BASE_FRAME, BOX_CENTER, BOX_ID, BOX_SIZE, END_EFFECTOR_LINK,
    GRASP_GRIPPER, GRIPPER_JOINT, GRIPPER_OPEN, GRIPPER_ORIENTATION,
    HOME_JOINTS, JOINT_TOLERANCE, LIFT_HEIGHT, ORIENTATION_TOLERANCE,
    PRE_GRASP_CLEARANCE, TOOL_LENGTH, TOUCH_LINKS, grasp_hand_position,
    lift_position, make_attached_object, make_box_object,
    make_joint_constraints, make_orientation_constraint,
    make_position_constraint, pre_grasp_hand_position, tcp_to_hand_position)
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive


def test_home_joints_match_panda_initial_positions():
    """Home 姿态必须是 Panda 官方 initial_positions.yaml 里的 7 个关节角."""
    assert HOME_JOINTS == (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
    assert len(HOME_JOINTS) == len(ARM_JOINTS) == 7


def test_arm_joint_names_are_panda_joints_1_to_7():
    """关节名必须按 panda_joint1..7 顺序排列."""
    assert ARM_JOINTS == tuple('panda_joint%d' % i for i in range(1, 8))


def test_box_is_five_centimeter_cube_at_requested_center():
    """箱子必须是 5 cm 方块，中心在 (0.5, 0, 0.25)."""
    assert BOX_SIZE == 0.05
    assert BOX_CENTER == (0.5, 0.0, 0.25)


def test_box_object_message_matches_box_constants():
    """加入场景的 CollisionObject 必须是方块、ADD、位姿正确."""
    obj = make_box_object()
    assert obj.id == BOX_ID
    assert obj.header.frame_id == BASE_FRAME == 'panda_link0'
    assert obj.operation == CollisionObject.ADD
    assert len(obj.primitives) == len(obj.primitive_poses) == 1

    primitive = obj.primitives[0]
    assert primitive.type == SolidPrimitive.BOX
    assert list(primitive.dimensions) == [BOX_SIZE] * 3

    center = obj.primitive_poses[0].position
    assert (center.x, center.y, center.z) == BOX_CENTER


def test_tcp_to_hand_position_offsets_only_z():
    """panda_hand 在抓取点上方 TOOL_LENGTH 处."""
    assert TOOL_LENGTH == 0.1122
    hand = tcp_to_hand_position((0.5, 0.0, 0.25))
    assert hand == (0.5, 0.0, 0.25 + TOOL_LENGTH)


def test_grasp_hand_position_puts_fingertips_at_box_center():
    """抓取时指尖必须停在箱心，而不是让连杆插进箱子里."""
    hand = grasp_hand_position()
    fingertip_z = hand[2] - TOOL_LENGTH
    assert abs(fingertip_z - BOX_CENTER[2]) < 1e-9


def test_pre_grasp_is_five_centimeters_above_box_center():
    """预抓取时指尖停在箱心正上方 5 cm."""
    hand = pre_grasp_hand_position()
    fingertip_z = hand[2] - TOOL_LENGTH
    assert abs(fingertip_z - (BOX_CENTER[2] + PRE_GRASP_CLEARANCE)) < 1e-9
    assert abs(PRE_GRASP_CLEARANCE - 0.05) < 1e-9


def test_attach_and_detach_messages_use_hand_and_fingers():
    """挂载/卸载必须作用在 panda_hand 上，并放开手指与箱子的碰撞."""

    def check(operation):
        message = make_attached_object(operation=operation)
        assert message.link_name == END_EFFECTOR_LINK == 'panda_hand'
        assert message.object.id == BOX_ID
        assert message.object.operation == operation
        assert tuple(message.touch_links) == TOUCH_LINKS
        assert 'panda_leftfinger' in TOUCH_LINKS
        assert 'panda_rightfinger' in TOUCH_LINKS

    check(CollisionObject.ADD)
    check(CollisionObject.REMOVE)


def test_joint_constraints_cover_every_joint():
    """每个关节都要有一条容差与权重正确的约束."""
    constraints = make_joint_constraints(ARM_JOINTS, HOME_JOINTS)
    assert [c.joint_name for c in constraints] == list(ARM_JOINTS)
    for constraint, expected in zip(constraints, HOME_JOINTS):
        assert constraint.position == expected
        assert constraint.tolerance_above == JOINT_TOLERANCE
        assert constraint.tolerance_below == JOINT_TOLERANCE
        assert constraint.weight == 1.0


def test_position_constraint_uses_sphere_at_box_center():
    """位置约束应是落在箱心上的小球，且绑定末端连杆."""
    constraint = make_position_constraint(END_EFFECTOR_LINK, BOX_CENTER)
    assert constraint.header.frame_id == BASE_FRAME == 'panda_link0'
    assert constraint.link_name == END_EFFECTOR_LINK == 'panda_hand'

    region = constraint.constraint_region
    assert len(region.primitives) == len(region.primitive_poses) == 1
    assert region.primitives[0].type == region.primitives[0].SPHERE

    center = region.primitive_poses[0].position
    assert (center.x, center.y, center.z) == BOX_CENTER


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


def test_gripper_stops_at_the_box_faces():
    """闭合位置等于箱子半边宽，手指正好合到箱子两侧."""
    assert GRASP_GRIPPER == BOX_SIZE / 2
    assert GRIPPER_OPEN == 0.04
    assert 0.0 < GRASP_GRIPPER < GRIPPER_OPEN


def test_grasp_constraint_targets_finger_joint():
    """夹爪指令必须落在 panda_finger_joint1 上（hand 组实际驱动的关节）."""
    constraints = make_joint_constraints([GRIPPER_JOINT], [GRASP_GRIPPER])
    assert len(constraints) == 1
    assert constraints[0].joint_name == 'panda_finger_joint1'
    assert constraints[0].position == GRASP_GRIPPER


def test_lift_position_raises_only_z():
    """抓取后只在 z 轴抬高 0.1 m，x/y 不变."""
    grasp = grasp_hand_position()
    lifted = lift_position(grasp)
    assert lifted == (grasp[0], grasp[1], grasp[2] + LIFT_HEIGHT)
    assert abs(lifted[2] - (BOX_CENTER[2] + TOOL_LENGTH + LIFT_HEIGHT)) < 1e-9
