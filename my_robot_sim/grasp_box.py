"""
The grasp target box: shared configuration and Gazebo SDF builders.

``config/grasp_box.yaml`` is the single source of truth. The launch file
spawns the cube and the pedestal it rests on in Gazebo with the SDF built
here, and ``mobile_manipulation_node`` uses the same numbers for its MoveIt
targets and planning-scene objects, so Gazebo, MoveIt and RViz cannot
disagree about where the box is.

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
    'pedestal_size',
    'friction',
    'tcp_offset',
    'approach_clearance',
    'squeeze_effort',
    'lift_height',
    'lift_velocity',
)

POSITIVE_KEYS = (
    'size', 'mass', 'friction', 'pedestal_size', 'squeeze_effort',
    'lift_velocity')
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
    return config


def box_pose(config):
    """Cube centre in the world frame."""
    return (config['x'], config['y'], config['z'])


def pedestal_height(config):
    """Pedestal height so its top face touches the cube's bottom face."""
    return config['z'] - config['size'] / 2.0


def pedestal_pose(config):
    """Pedestal centre in the world frame."""
    return (config['x'], config['y'], pedestal_height(config) / 2.0)


def pedestal_dimensions(config):
    """Pedestal (x, y, z) edge lengths."""
    return (config['pedestal_size'], config['pedestal_size'],
            pedestal_height(config))


def box_sdf(config):
    """
    Return the SDF for the dynamic grasp cube.

    The cube is spawned at the configured pose, so the model itself carries
    no pose. Orange makes it obvious in gzclient, and the high friction plus
    the small mass are what let the fingers actually carry it.
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
            <name>Gazebo/Orange</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""" % (mass, inertia, inertia, inertia, dimensions, friction, friction,
       dimensions)


def pedestal_sdf(config):
    """Return the SDF for the static pedestal that holds the cube up."""
    x_size, y_size, z_size = pedestal_dimensions(config)
    dimensions = '%g %g %g' % (x_size, y_size, z_size)
    return """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="grasp_pedestal">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>%s</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>%s</size></box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Grey</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""" % (dimensions, dimensions)
