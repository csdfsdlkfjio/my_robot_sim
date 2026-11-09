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


def test_ground_truth_odometry_plugin_tracks_base_link(robot):
    """Grasp geometry needs the true base pose, not slipping wheel odometry."""
    plugins = {
        plugin.get('name'): plugin for plugin in robot.iter('plugin')}
    plugin = plugins['ground_truth_odometry']
    assert plugin.get('filename') == 'libgazebo_ros_p3d.so'
    assert plugin.find('body_name').text.strip() == 'base_link'
    assert plugin.find('frame_name').text.strip() == 'world'
    # The Humble plugin hardcodes its publisher to "odom", so the topic must
    # be remapped away from the differential-drive /odom.
    assert plugin.find('ros/remapping').text.strip() == \
        'odom:=ground_truth/odom'
    assert float(plugin.find('update_rate').text) > 0.0
    assert float(plugin.find('gaussian_noise').text) == 0.0


def test_robot_model_loads_the_grasp_attach_plugin(robot):
    """
    A URDF plugin is loaded by file name, not by package path.

    The launch file puts my_robot_grasp_attach's lib directory on
    GAZEBO_PLUGIN_PATH; this test pins the file name the URDF asks for. Without
    the plugin the cube stays on its pedestal when the arm lifts.
    """
    plugins = {
        plugin.get('name'): plugin for plugin in robot.iter('plugin')}
    plugin = plugins['grasp_attach']
    assert plugin.get('filename') == 'libgazebo_grasp_attach.so'


def test_wrist_camera_is_mounted_on_the_hand(robot):
    """
    The camera must look along the fingers (down while grasping).

    Gazebo renders a camera along its link's +x axis, so the mount has to
    pitch the link by -90 degrees: the hand's +z axis is the finger direction.
    The optical frame is the conventional ROS camera frame (z forward, x
    right, y down) and is what the images are stamped with. Both frames are
    fixed joints in the URDF, so robot_state_publisher publishes them.
    """
    joints = named(robot, 'joint')
    mount = joints['camera_joint']
    assert mount.get('type') == 'fixed'
    assert mount.find('parent').get('link') == 'panda_hand'
    assert mount.find('child').get('link') == 'camera_link'
    origin = mount.find('origin')
    assert [float(value) for value in origin.get('xyz').split()] == \
        pytest.approx([0.0, 0.05, 0.0])
    assert [float(value) for value in origin.get('rpy').split()] == \
        pytest.approx([0.0, -1.5707963, 0.0])

    optical = joints['camera_optical_joint']
    assert optical.get('type') == 'fixed'
    assert optical.find('parent').get('link') == 'camera_link'
    assert optical.find('child').get('link') == 'camera_optical_link'
    assert [float(value)
            for value in optical.find('origin').get('rpy').split()] == \
        pytest.approx([-1.5707963, 0.0, -1.5707963])

    links = named(robot, 'link')
    # The camera body must not collide with the cube the gripper picks up.
    assert links['camera_link'].find('collision') is None
    assert links['camera_link'].find('inertial') is not None
    references = {
        block.get('reference'): block for block in robot.findall('gazebo')}
    assert references['camera_link'].findtext('material').strip() == \
        'Gazebo/DarkGrey'


def test_camera_sensor_reaches_the_gazebo_model(urdf_text):
    """
    The sensor must survive the URDF->SDF conversion with its plugin intact.

    The camera link is fixed to panda_hand, which Gazebo already lumps into
    panda_link7; the converter moves the sensor along with the lumped visual,
    and drops it entirely if the <gazebo reference> block is malformed.
    """
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

    sensors = [sensor for sensor in root.iter('sensor')
               if sensor.get('name') == 'arm_camera']
    assert len(sensors) == 1
    sensor = sensors[0]
    assert sensor.get('type') == 'camera'
    assert float(sensor.findtext('update_rate')) == pytest.approx(30.0)
    image = sensor.find('camera/image')
    assert int(image.findtext('width')) == 640
    assert int(image.findtext('height')) == 480
    assert image.findtext('format').strip() == 'R8G8B8'
    plugin = sensor.find('plugin')
    assert plugin.get('filename') == 'libgazebo_ros_camera.so'
    # camera_name prefixes the topics: /camera/image_raw and
    # /camera/camera_info. frame_name must match the optical frame that
    # robot_state_publisher publishes, or the image TF is disconnected.
    assert plugin.findtext('camera_name').strip() == 'camera'
    assert plugin.findtext('frame_name').strip() == 'camera_optical_link'


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


def test_fingers_expose_position_and_effort_commands(robot):
    """
    The hand switches controllers between opening and squeezing.

    Position is needed for opening/pre-close, but a position command is a
    teleport in gazebo_ros2_control: the pad would be placed inside the cube
    and the contact would eject it, giving no steady normal force. The effort
    interface is what makes the squeeze physical. Both therefore have to reach
    the hardware through the URDF.
    """
    system = robot.find('ros2_control')
    joints = {
        joint.get('name'): joint for joint in system.findall('joint')}
    for name in ('panda_finger_joint1', 'panda_finger_joint2'):
        interfaces = {
            interface.get('name')
            for interface in joints[name].findall('command_interface')}
        assert interfaces == {'position', 'effort'}, name
    for name in ('panda_joint1', 'panda_joint7'):
        interfaces = {
            interface.get('name')
            for interface in joints[name].findall('command_interface')}
        assert interfaces == {'position'}, name


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


def test_fingers_use_a_flat_box_pad_for_grasping(robot):
    """
    finger.stl's inner face is recessed above z = 0.036.

    A cube gripped at the hand origin only touches the last 1.8 cm of the
    mesh and slips out. The collision must therefore be the full-height box
    envelope, while the visual keeps the original DAE.

    Each pad must sit on its own side of the link origin: the finger link
    plane is the grasp face, and the 2.65 cm thick pad extends outward from
    it. A rotation cannot mirror a box (it is symmetric about its centre), so
    the right pad is mirrored by a negative y offset; a positive offset there
    would put the pad inside the grasp gap and through the cube.
    """
    links = named(robot, 'link')
    offsets = {'panda_leftfinger': 0.01315, 'panda_rightfinger': -0.01315}
    for name, offset in offsets.items():
        collision = links[name].find('collision')
        assert collision is not None, name
        box = collision.find('geometry/box')
        assert box is not None, '%s collision is not a box' % name
        assert collision.find('geometry/mesh') is None, name
        assert [float(value) for value in box.get('size').split()] == \
            pytest.approx([0.040, 0.0265, 0.0538])
        origin = collision.find('origin')
        assert float(origin.get('xyz').split()[1]) == pytest.approx(offset)
        assert [float(value) for value in origin.get('rpy').split()] == \
            pytest.approx([0.0, 0.0, 0.0])
        # The visual must keep the real finger shape.
        assert links[name].find('visual/geometry/mesh') is not None, name


def test_fingers_carry_grasp_contact_parameters_into_the_sdf(urdf_text):
    """
    The physical grasp is held by friction alone.

    The finger <gazebo reference> blocks must therefore reach the SDF as
    high-friction, stiff contacts. A soft contact (kp = 1e6) was measured to
    let the pads slide up the cube with no vertical force at all, so the
    stiffness matters as much as mu. The values mirror Gazebo's
    friction_demo.world.
    """
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
    for name in ('panda_leftfinger', 'panda_rightfinger'):
        link = next(item for item in root.iter('link')
                    if item.get('name') == name)
        collision = link.find('collision')
        assert float(collision.findtext(
            'surface/friction/ode/mu')) >= 1.0, name
        assert float(collision.findtext(
            'surface/friction/ode/mu2')) >= 1.0, name
        # Match the friction demo: a stiff contact keeps the friction
        # constraint from slipping.
        assert float(collision.findtext(
            'surface/contact/ode/kp')) >= 1e9, name
        assert float(collision.findtext(
            'surface/contact/ode/max_vel')) >= 100.0, name
        assert float(collision.findtext(
            'surface/contact/ode/min_depth')) > 0.0, name


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
