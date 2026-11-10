"""
The grasp target box: shared configuration and Gazebo SDF builders.

``config/grasp_box.yaml`` describes the cube and the long table it rests on.
The launch file spawns the cube in Gazebo with the SDF built here; the table
itself is a static model in ``worlds/demo.world`` so SLAM and Nav2 can see it.
``mobile_manipulation_node`` uses the same numbers for its MoveIt targets and
planning-scene objects, so Gazebo, MoveIt and RViz cannot disagree about the
table or the cube. A test compares the YAML table with the world model.

Poses are in the Gazebo world frame. The robot spawns at the world origin.
``mobile_manipulation_node`` reads the live base pose from Gazebo's
ground-truth odometry, not from the differential-drive /odom, which is wheel
odometry and drifts by centimetres over the drive.
"""

import yaml

# Distance from the panda_hand frame origin to the front face of the palm,
# measured on hand.stl. The hand +z axis points along the fingers, so in a
# top grasp the palm face sits this far "below" the hand origin.
PALM_DEPTH = 0.066

# The finger joints are limited to this effort in the URDF; a larger squeeze
# command would simply saturate.
MAX_FINGER_EFFORT = 20.0

CONFIG_KEYS = (
    'size',
    'mass',
    'x',
    'y',
    'z',
    'table_x',
    'table_y',
    'table_size_x',
    'table_size_y',
    'table_height',
    'friction',
    'tcp_offset',
    'approach_clearance',
    'squeeze_effort',
    'lift_height',
    'lift_velocity',
)

POSITIVE_KEYS = (
    'size', 'mass', 'friction', 'table_size_x', 'table_size_y',
    'table_height', 'squeeze_effort', 'lift_velocity')
NON_NEGATIVE_KEYS = ('approach_clearance', 'lift_height')


def load_config(path):
    """Read and validate the grasp box configuration."""
    with open(path, encoding='utf-8') as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or 'grasp_box' not in document:
        raise ValueError('%s has no grasp_box section' % path)
    section = document['grasp_box']
    missing = [key for key in CONFIG_KEYS if key not in section]
    if missing:
        raise ValueError('%s is missing %s' % (path, ', '.join(missing)))

    config = {key: float(section[key]) for key in CONFIG_KEYS}
    for key in POSITIVE_KEYS:
        if config[key] <= 0.0:
            raise ValueError('%s: %s must be positive' % (path, key))
    for key in NON_NEGATIVE_KEYS:
        if config[key] < 0.0:
            raise ValueError('%s: %s must not be negative' % (path, key))

    clearance = PALM_DEPTH + config['size'] / 2.0
    if config['tcp_offset'] <= clearance:
        raise ValueError(
            '%s: tcp_offset (%g) must exceed the palm depth plus half the '
            'cube (%g), otherwise the palm collides with the cube'
            % (path, config['tcp_offset'], clearance))
    if config['squeeze_effort'] > MAX_FINGER_EFFORT:
        raise ValueError(
            '%s: squeeze_effort (%g) exceeds the %g N finger effort limit'
            % (path, config['squeeze_effort'], MAX_FINGER_EFFORT))
    if config['lift_velocity'] > 1.0:
        raise ValueError(
            '%s: lift_velocity (%g) is a MoveIt velocity scaling factor, so '
            'it must not exceed 1.0' % (path, config['lift_velocity']))

    table_top = config['table_height']
    cube_bottom = config['z'] - config['size'] / 2.0
    if abs(table_top - cube_bottom) > 1e-3:
        raise ValueError(
            '%s: the cube bottom (%g) must rest on the table top (%g); '
            'set z = table_height + size / 2'
            % (path, cube_bottom, table_top))
    if abs(config['x'] - config['table_x']) > config['table_size_x'] / 2.0:
        raise ValueError(
            '%s: the cube x (%g) is not above the table top'
            % (path, config['x']))
    if abs(config['y'] - config['table_y']) > config['table_size_y'] / 2.0:
        raise ValueError(
            '%s: the cube y (%g) is not above the table top'
            % (path, config['y']))
    return config


def box_pose(config):
    """Cube centre in the world frame."""
    return (config['x'], config['y'], config['z'])


def table_pose(config):
    """Table centre in the world frame (the table is fixed, the cube is not)."""
    return (config['table_x'], config['table_y'],
            config['table_height'] / 2.0)


def table_dimensions(config):
    """Table (x, y, z) edge lengths; the long axis is Y."""
    return (config['table_size_x'], config['table_size_y'],
            config['table_height'])


def box_sdf(config):
    """
    Return the SDF for the dynamic grasp cube.

    The cube is spawned at the configured pose, so the model itself carries
    no pose. Red is what the wrist camera's colour threshold looks for (see
    visual_grasp.py), and the high friction plus the small mass are what let
    the fingers actually carry it.
    """
    size = config['size']
    mass = config['mass']
    friction = config['friction']
    # Solid cube: I = m * (s^2 + s^2) / 12 about every axis.
    inertia = mass * (size * size + size * size) / 12.0
    dimensions = '%g %g %g' % (size, size, size)
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="grasp_box">
    <allow_auto_disable>false</allow_auto_disable>
    <link name="link">
      <inertial>
        <mass>%.12g</mass>
        <inertia>
          <ixx>%.12g</ixx>
          <iyy>%.12g</iyy>
          <izz>%.12g</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>%s</size></box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>%g</mu>
              <mu2>%g</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>%s</size></box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Red</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""" % (mass, inertia, inertia, inertia, dimensions, friction, friction,
       dimensions)
