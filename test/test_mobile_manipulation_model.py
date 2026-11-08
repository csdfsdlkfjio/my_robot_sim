"""
Static checks for the merged car + Panda model.

These tests run xacro on the installed description and then verify the two
control paths (Gazebo diff-drive plugin and ros2_control) own disjoint joints.
"""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest
import xacro

XACRO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'urdf',
    'mobile_manipulator.urdf.xacro'))

PANDA_JOINTS = {'panda_joint%d' % index for index in range(1, 8)}
HAND_JOINTS = {'panda_finger_joint1', 'panda_finger_joint2'}
WHEEL_JOINTS = {'left_wheel_joint', 'right_wheel_joint'}


@pytest.fixture(scope='module')
def robot():
    """Process the merged xacro once for all tests."""
    return ET.fromstring(xacro.process_file(XACRO).toxml())


@pytest.fixture(scope='module')
def urdf_text():
    """Return the processed URDF as text."""
    return xacro.process_file(XACRO).toxml()


def named(elements, tag):
    """Index child elements by their ``name`` attribute."""
    return {element.get('name'): element for element in elements.findall(tag)}


def test_merged_robot_has_single_root(robot, urdf_text):
    """base_link must be the only root, so Gazebo and MoveIt see one tree."""
    with tempfile.NamedTemporaryFile(
            'w', suffix='.urdf', delete=False) as handle:
        handle.write(urdf_text)
        path = handle.name
    try:
        result = subprocess.run(
            ['check_urdf', path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Two root links' not in result.stdout
    assert 'base_link' in result.stdout
    assert robot.get('name') == 'mobile_manipulator'


def test_arm_is_mounted_on_the_car_roof(robot):
    """Panda link0 is fixed to the mounting plate on base_link."""
    joints = named(robot, 'joint')
    mount = joints['arm_mount_joint']
    assert mount.get('type') == 'fixed'
    assert mount.find('parent').get('link') == 'base_link'
    assert mount.find('child').get('link') == 'arm_mount_link'

    panda_joint = joints['panda_base_joint']
    assert panda_joint.get('type') == 'fixed'
    assert panda_joint.find('parent').get('link') == 'arm_mount_link'
    assert panda_joint.find('child').get('link') == 'panda_link0'

    links = named(robot, 'link')
    for link in ('arm_mount_link', 'ballast_link', 'panda_link0',
                 'panda_hand', 'panda_leftfinger', 'panda_rightfinger'):
        assert link in links, 'missing link %s' % link


def test_every_panda_link_has_positive_inertia(robot):
    """
    Gazebo Classic silently drops links without <inertial>.

    panda_link8 is the massless tool frame between link7 and the hand; it is
    intentionally left without inertia and is lumped into link7 by Gazebo.
    """
    links = named(robot, 'link')
    panda_links = [name for name in links if name.startswith('panda_')]
    assert len(panda_links) == 12
    assert links['panda_link8'].find('inertial') is None
    for name in panda_links:
        if name == 'panda_link8':
            continue
        inertial = links[name].find('inertial')
        assert inertial is not None, '%s has no inertial' % name
        assert float(inertial.find('mass').get('value')) > 0.0
        inertia = inertial.find('inertia')
        for axis in ('ixx', 'iyy', 'izz'):
            assert float(inertia.get(axis)) > 0.0


def test_diff_drive_plugin_owns_only_wheels(robot):
    """The differential-drive plugin must still drive the two wheel joints."""
    plugins = {
        plugin.get('name'): plugin for plugin in robot.iter('plugin')
    }
    diff_drive = plugins['diff_drive']
    assert diff_drive.get('filename') == 'libgazebo_ros_diff_drive.so'
    assert diff_drive.find('left_joint').text.strip() == 'left_wheel_joint'
    assert diff_drive.find('right_joint').text.strip() == 'right_wheel_joint'
    assert diff_drive.find('publish_odom').text.strip() == 'true'
    assert diff_drive.find('publish_odom_tf').text.strip() == 'true'


def test_ros2_control_owns_only_the_arm(robot):
    """Every <ros2_control> joint must be a Panda joint, never a wheel."""
    systems = robot.findall('ros2_control')
    assert len(systems) == 1
    system = systems[0]
    hardware = system.find('hardware/plugin').text.strip()
    assert hardware == 'gazebo_ros2_control/GazeboSystem'

    controlled = {
        joint.get('name') for joint in system.findall('joint')}
    assert controlled == PANDA_JOINTS | HAND_JOINTS


def test_control_paths_are_disjoint(robot):
    """No joint may be driven by both the plugins and ros2_control."""
    diff_drive = next(
        plugin for plugin in robot.iter('plugin')
        if plugin.get('name') == 'diff_drive')
    plugin_joints = {
        diff_drive.find('left_joint').text.strip(),
        diff_drive.find('right_joint').text.strip(),
    }
    system = robot.find('ros2_control')
    controlled = {joint.get('name') for joint in system.findall('joint')}
    assert plugin_joints & controlled == set()
    assert plugin_joints == WHEEL_JOINTS


def test_gazebo_ros2_control_plugin_is_configured(robot):
    """The controller manager inside Gazebo needs the controller YAML."""
    plugin = next(
        item for item in robot.iter('plugin')
        if item.get('filename') == 'libgazebo_ros2_control.so')
    parameters = plugin.find('parameters')
    assert parameters is not None
    assert parameters.text.strip().endswith(
        'config/mobile_manip_controllers.yaml')
    assert os.path.isabs(parameters.text.strip())
    assert plugin.find('robot_param_node').text.strip() == \
        'mobile_robot_state_publisher'


def test_mesh_uris_are_absolute_file_urls(robot):
    """
    Gazebo turns package:// into model://, which is not on its model path.

    An unresolvable collision mesh makes gzserver segfault shortly after the
    model is spawned, so every mesh must already carry an absolute file URI.
    """
    meshes = [mesh.get('filename') for mesh in robot.iter('mesh')]
    assert meshes
    for filename in meshes:
        assert filename.startswith('file:///'), filename


def test_every_panda_link_gets_a_gazebo_material(robot):
    """
    Gazebo Classic does not draw mesh visuals that have no material.

    URDF <material> colours never reach the SDF, so the material has to be
    passed through <gazebo reference> blocks: without them the arm is
    simulated but invisible in gzclient.
    """
    references = {
        block.get('reference'): block
        for block in robot.findall('gazebo')
        if block.get('reference') is not None
        and block.find('material') is not None
    }
    panda_links = ['panda_link%d' % index for index in range(8)]
    panda_links += ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']
    for link in panda_links:
        assert link in references, '%s has no gazebo material' % link
        assert references[link].findtext('material').strip() == 'Gazebo/White'


def test_gazebo_sdf_conversion_keeps_the_panda_material(urdf_text):
    """The gazebo reference material must survive the conversion to SDF."""
    with tempfile.NamedTemporaryFile(
            'w', suffix='.urdf', delete=False) as handle:
        handle.write(urdf_text)
        path = handle.name
    try:
        result = subprocess.run(
            ['gz', 'sdf', '-p', path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    assert result.returncode == 0, result.stderr
    root = ET.fromstring(result.stdout)
    link = next(item for item in root.iter('link')
                if item.get('name') == 'panda_link1')
    visual = link.find('visual')
    name = visual.find('material/script/name')
    assert name is not None and name.text.strip() == 'Gazebo/White'


def test_gazebo_sdf_conversion_keeps_all_links(urdf_text):
    """The gz sdf conversion must keep a non-empty model with all links."""
    with tempfile.NamedTemporaryFile(
            'w', suffix='.urdf', delete=False) as handle:
        handle.write(urdf_text)
        path = handle.name
    try:
        result = subprocess.run(
            ['gz', 'sdf', '-p', path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    assert result.returncode == 0, result.stderr
    root = ET.fromstring(result.stdout)
    model = root.find('model')
    assert model is not None
    sdf_links = model.findall('link')
    assert len(sdf_links) >= 10
    assert any(link.get('name') == 'base_link' for link in sdf_links)
    assert any(link.get('name').startswith('panda_link1')
               for link in sdf_links)


def test_srdf_has_no_second_root():
    """The standalone Panda SRDF's world->panda_link0 joint must be removed."""
    srdf = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'config', 'mobile_manip.srdf'))
    tree = ET.parse(srdf).getroot()
    assert tree.find('virtual_joint') is None


def test_no_mimic_joints_in_the_urdf(robot):
    """The URDF must not use mimic: MoveIt would drop it from planning."""
    assert robot.findall('.//mimic') == []


def test_srdf_hand_group_has_two_active_fingers():
    """Gazebo cannot simulate <mimic>, so both fingers are planned joints."""
    srdf = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'config', 'mobile_manip.srdf'))
    tree = ET.parse(srdf).getroot()
    hand = next(group for group in tree.findall('group')
                if group.get('name') == 'hand')
    joints = {joint.get('name') for joint in hand.findall('joint')}
    assert joints == HAND_JOINTS
    assert hand.find('passive_joint') is None
