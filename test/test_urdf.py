"""URDF 关节与关节状态发布配置的静态检查."""

import os
import xml.etree.ElementTree as ET

URDF = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'my_car.urdf')
WHEEL_JOINTS = {'left_wheel_joint', 'right_wheel_joint'}


def load_urdf():
    """解析 URDF 并返回根节点."""
    return ET.parse(URDF).getroot()


def test_wheel_joints_are_continuous():
    """两个轮子必须是 continuous 关节."""
    root = load_urdf()
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    for name in WHEEL_JOINTS:
        assert name in joints, f'缺少关节 {name}'
        assert joints[name].get('type') == 'continuous'


def test_gazebo_publishes_wheel_joint_states():
    """必须有插件发布两个轮子的 /joint_states，否则轮子没有 TF."""
    root = load_urdf()
    plugins = [
        plugin for plugin in root.iter('plugin')
        if plugin.get('filename') == 'libgazebo_ros_joint_state_publisher.so'
    ]
    assert len(plugins) == 1, '应配置一个 gazebo 关节状态发布插件'
    names = {j.text.strip() for j in plugins[0].findall('joint_name')}
    assert names == WHEEL_JOINTS
    remaps = {r.text.strip() for r in plugins[0].findall('ros/remapping')}
    assert '~/out:=joint_states' in remaps
