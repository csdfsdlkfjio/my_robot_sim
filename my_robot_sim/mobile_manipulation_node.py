"""
Mobile-manipulation state machine that really picks up a cube.

Sequence:
    1. wait for /odom, /joint_states and MoveIt's /move_action;
    2. send the Panda home pose straight to its JointTrajectoryController
       (the Gazebo spawn pose is slightly outside the Panda joint limits and
       self-colliding, which MoveIt would reject as a start state);
    3. open both gripper fingers with MoveIt;
    4. drive forward ``drive_distance`` metres, checked against /odom;
    5. publish zero velocity and let the base settle;
    6. put the grasp cube and its pedestal into MoveIt's planning scene at
       their real pose (RViz shows them, MoveIt avoids them);
    7. MoveIt to 5 cm above the cube's top face, then straight down until the
       grasp point is at the cube centre;
    8. close both fingers until they squeeze the cube;
    9. weld the cube to the Gazebo hand link, panda_link7, through a fixed
       joint created by the my_robot_grasp_attach plugin (Gazebo has no
       panda_hand link: it is lumped into link7), attach it in MoveIt's
       planning scene (AttachedCollisionObject) and lift the arm 10 cm.

Arm motion is executed by the two gazebo_ros2_control
JointTrajectoryControllers (homing directly, everything else through
MoveIt). The base is still driven by the Gazebo differential-drive plugin, so
the two control paths never touch the same joint.

The fingers are special: they are opened and pre-closed by their position
JointTrajectoryController, then switched to a constant-effort
JointGroupEffortController for the squeeze. A position command is a teleport
in gazebo_ros2_control and produces no steady normal force, so a
position-only grasp cannot carry the cube. The effort press still does not
lift it: measured in Gazebo the pads slide up the cube's faces while the
cube stays put, whatever the force or cube mass. The physical lift therefore
comes from the my_robot_grasp_attach plugin, which is told to create a fixed
joint between the cube and panda_hand once the fingers are closed. The
AttachedCollisionObject is the MoveIt/RViz counterpart: it is what makes the
planner and RViz carry the cube with the hand.

The box pose is configured in config/grasp_box.yaml (the launch file spawns
the same cube and pedestal in Gazebo) and is expressed in the Gazebo world
frame. The live base pose comes from libgazebo_ros_p3d's ground-truth
odometry (/ground_truth/odom), not from the differential-drive /odom: the
latter is wheel odometry and over a 1 m drive it ended up ~3 cm and ~2.5
degrees away from where the robot really was, which is enough to make the
closed pads pinch only the front edge of the 5 cm cube instead of its faces.
If the ground-truth topic never arrives the node falls back to the odom TF,
so unit tests and launch files without the plugin still work.
"""

import math
import os
import time

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion, Twist
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float64MultiArray, String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from my_robot_sim.grasp_box import (
    box_pose,
    load_config,
    pedestal_dimensions,
    pedestal_pose,
)

ARM_JOINTS = tuple('panda_joint%d' % index for index in range(1, 8))
HAND_JOINTS = ('panda_finger_joint1', 'panda_finger_joint2')

# Same home pose as the Panda MoveIt configuration.
HOME_JOINTS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

ARM_GROUP = 'panda_arm'
HAND_GROUP = 'hand'
ARM_BASE_FRAME = 'panda_link0'
END_EFFECTOR_LINK = 'panda_hand'
ODOM_FRAME = 'odom'

# Name of the hand link inside Gazebo. The URDF->SDF conversion lumps every
# link behind a fixed joint into its parent, and panda_hand is fixed to
# panda_link7 (through the massless panda_link8). gz model -i lists
# panda_link1..7 but no panda_hand, so the Gazebo weld has to name the link
# that really exists. MoveIt still uses panda_hand.
GAZEBO_HAND_LINK = 'panda_link7'

# Ground-truth base pose from libgazebo_ros_p3d (see mobile_manipulator.
# urdf.xacro). Wheel odometry drifts; this does not.
GROUND_TRUTH_TOPIC = '/ground_truth/odom'
GROUND_TRUTH_TIMEOUT = 5.0

# base_link -> panda_link0 is a pure translation: arm_mount_joint moves
# (-0.04, 0, 0.03) and panda_base_joint a further (0, 0, 0.005), with no
# rotation on either. Coordinates in the base frame therefore turn into
# panda_link0 coordinates by adding this offset.
ARM_MOUNT_OFFSET = (-0.04, 0.0, 0.035)

# The homing motion bypasses MoveIt and talks to the ros2_control JTC directly.
ARM_TRAJECTORY_ACTION = '/panda_arm_controller/follow_joint_trajectory'
HOME_DURATION = 4.0  # s
HOME_TIMEOUT = 30.0  # s

# Straight-line Cartesian motions (the grasp descend and the lift) go through
# MoveIt's Cartesian planner. A free-space RRT path from the pre-grasp to the
# grasp is short but need not be straight, so it can clip the cube with the
# open fingers even though both end points are valid.
CARTESIAN_SERVICE = '/compute_cartesian_path'
EXECUTE_ACTION = '/execute_trajectory'
CARTESIAN_MAX_STEP = 0.005  # m between consecutive path points
CARTESIAN_MIN_FRACTION = 0.99

# Orientation (x=1, w=0) is a 180 degree turn about x: the hand +z axis, which
# runs along the fingers, points at the floor.
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)

# Panda finger travel is 0..0.04 m (0 = fingers together, gap = 2 * joint).
GRIPPER_OPEN = 0.04

# The gripper is opened and pre-closed by the position JTC, then switched to
# a dedicated effort controller for the squeeze. A position command is a
# teleport in gazebo_ros2_control and generates no steady normal force, and a
# position-derived PID is unstable on the 15 g fingers.
HAND_POSITION_CONTROLLER = 'panda_hand_controller'
HAND_EFFORT_CONTROLLER = 'panda_hand_effort_controller'
HAND_EFFORT_TOPIC = '/panda_hand_effort_controller/commands'
SWITCH_CONTROLLER_SERVICE = '/controller_manager/switch_controller'
PRE_CLOSE_CLEARANCE = 0.005  # m between pad and cube before the force press
SQUEEZE_SETTLE_SECONDS = 0.8
SQUEEZE_CONTACT_TOLERANCE = 0.003  # m; pads must rest on the cube faces

DRIVE_DISTANCE = 1.0
FORWARD_SPEED = 0.25
STOP_PUBLISH_SECONDS = 1.0
SETTLE_SECONDS = 0.5
DRIVE_TIMEOUT = 30.0
MOVE_TIMEOUT = 90.0
TF_TIMEOUT = 15.0
SCENE_SETTLE_SECONDS = 0.5

POSITION_TOLERANCE = 0.01  # m
ORIENTATION_TOLERANCE = 0.05  # rad, about 2.9 degrees
JOINT_TOLERANCE = 0.01  # rad or m

BOX_OBJECT_ID = 'grasp_box'
PEDESTAL_OBJECT_ID = 'grasp_pedestal'

# Gazebo entity names. The task node tells the my_robot_grasp_attach plugin to
# weld these two links together once the fingers squeeze the cube.
ROBOT_MODEL_NAME = 'mobile_manipulator'
BOX_MODEL_NAME = 'grasp_box'
BOX_LINK_NAME = 'link'
GRASP_ATTACH_TOPIC = '/grasp_attach'
GRASP_ATTACH_STATUS_TOPIC = '/grasp_attach/status'
ATTACH_TIMEOUT = 5.0
ATTACH_REPUBLISH_SECONDS = 0.2

# The attached cube may touch the gripper that holds it; MoveIt must not
# report those contacts as self-collisions.
TOUCH_LINKS = ('panda_hand', 'panda_leftfinger', 'panda_rightfinger')


def default_grasp_config_path():
    """Path of grasp_box.yaml, from the install or directly from the source."""
    try:
        share = get_package_share_directory('my_robot_sim')
    except LookupError:
        # Running straight from the source tree (plain pytest) without the
        # install prefix on AMENT_PREFIX_PATH.
        share = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.join(share, 'config', 'grasp_box.yaml')


def odom_distance(start_xy, current_xy):
    """Return the planar distance between two (x, y) points."""
    return math.hypot(current_xy[0] - start_xy[0],
                      current_xy[1] - start_xy[1])


def tcp_hand_position(box_position, tcp_offset, extra_height=0.0):
    """
    Hand origin that puts the grasp point ``extra_height`` above the box.

    With GRIPPER_DOWN the hand +z axis points at the floor, so the hand sits
    ``tcp_offset`` above the grasp point. The grasp point is the cube centre
    plus ``extra_height``.
    """
    return (
        float(box_position[0]),
        float(box_position[1]),
        float(box_position[2]) + float(extra_height) + float(tcp_offset),
    )


def pre_close_position(size, clearance=PRE_CLOSE_CLEARANCE):
    """
    Finger joint value that leaves ``clearance`` between pad and cube.

    The finger gap is twice the joint value, so the pads touch the cube at
    size / 2. Stopping just outside it keeps the position controller from
    teleporting a pad into the cube before the effort controller takes over.
    """
    return float(size) / 2.0 + float(clearance)


def finger_gap(positions):
    """
    Gap between the pads for the two finger joint values.

    Each finger joint value is one pad's displacement from the closed pose,
    so the pads' inner faces are ``sum(positions)`` apart.
    """
    return float(positions[0]) + float(positions[1])


def pads_hold_a_cube(positions, size, tolerance=SQUEEZE_CONTACT_TOLERANCE):
    """
    Return True when the pads closed on the cube instead of past it.

    The cube can sit a little off-centre between the pads: whichever finger
    touches first pushes the cube towards the other one, and friction stops it
    from sliding back. What must match the cube width is therefore the pad
    gap, not each finger separately. A missed grasp leaves the fingers fully
    closed and the gap near zero.
    """
    return abs(finger_gap(positions) - float(size)) <= 2.0 * tolerance


def make_effort_command(effort, joint_count=len(HAND_JOINTS)):
    """
    Build the constant-effort command that closes both fingers.

    Both finger joint axes point outward, so a negative effort presses the
    pads inward. The effort controller is open loop: the force is exactly the
    configured squeeze, independent of position.
    """
    message = Float64MultiArray()
    message.data = [-abs(float(effort))] * joint_count
    return message


def make_switch_request(activate, deactivate):
    """Build a strict controller swap for the finger command interfaces."""
    request = SwitchController.Request()
    request.activate_controllers = list(activate)
    request.deactivate_controllers = list(deactivate)
    request.strictness = SwitchController.Request.STRICT
    request.start_asap = False
    request.timeout.sec = 5
    return request


def make_joint_constraints(joint_names, positions, tolerance=JOINT_TOLERANCE):
    """Build one JointConstraint per joint."""
    return [
        JointConstraint(
            joint_name=name,
            position=float(position),
            tolerance_above=tolerance,
            tolerance_below=tolerance,
            weight=1.0,
        )
        for name, position in zip(joint_names, positions)
    ]


def quaternion_multiply(first, second):
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def inverse_quaternion(quaternion):
    """Conjugate of a unit (x, y, z, w) quaternion."""
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def rotate_vector(quaternion, vector):
    """Rotate ``vector`` by a unit (x, y, z, w) quaternion."""
    x, y, z, w = quaternion
    vx, vy, vz = (float(component) for component in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def make_pose(position, quaternion=GRIPPER_DOWN):
    """Build a Pose at ``position`` with the gripper-down orientation."""
    x, y, z, w = quaternion
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation = Quaternion(
        x=float(x), y=float(y), z=float(z), w=float(w))
    return pose


def make_cartesian_request(waypoints, link_name=END_EFFECTOR_LINK,
                           frame_id=ARM_BASE_FRAME,
                           max_step=CARTESIAN_MAX_STEP,
                           avoid_collisions=True,
                           velocity=0.3, acceleration=0.3):
    """Build a GetCartesianPath request; an empty start state means now."""
    request = GetCartesianPath.Request()
    request.header.frame_id = frame_id
    request.group_name = ARM_GROUP
    request.link_name = link_name
    request.waypoints = list(waypoints)
    request.max_step = float(max_step)
    request.jump_threshold = 0.0
    request.prismatic_jump_threshold = 0.0
    request.revolute_jump_threshold = 0.0
    request.avoid_collisions = bool(avoid_collisions)
    request.max_velocity_scaling_factor = float(velocity)
    request.max_acceleration_scaling_factor = float(acceleration)
    return request


def make_position_constraint(link_name, position,
                             frame_id=ARM_BASE_FRAME,
                             radius=POSITION_TOLERANCE):
    """Constrain ``link_name`` to a small sphere around ``position``."""
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[radius])
    sphere_pose = Pose()
    sphere_pose.position.x = float(position[0])
    sphere_pose.position.y = float(position[1])
    sphere_pose.position.z = float(position[2])
    sphere_pose.orientation.w = 1.0

    constraint = PositionConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.constraint_region = BoundingVolume(
        primitives=[sphere], primitive_poses=[sphere_pose])
    constraint.weight = 1.0
    return constraint


def make_orientation_constraint(link_name, quaternion=GRIPPER_DOWN,
                                frame_id=ARM_BASE_FRAME,
                                tolerance=ORIENTATION_TOLERANCE):
    """Constrain ``link_name`` to a fixed orientation."""
    x, y, z, w = quaternion
    constraint = OrientationConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.orientation = Quaternion(
        x=float(x), y=float(y), z=float(z), w=float(w))
    constraint.absolute_x_axis_tolerance = tolerance
    constraint.absolute_y_axis_tolerance = tolerance
    constraint.absolute_z_axis_tolerance = tolerance
    constraint.weight = 1.0
    return constraint


def make_move_goal(group_name, constraints, attempts=10,
                   planning_time=10.0, velocity=0.3, acceleration=0.3):
    """Wrap constraints into a MoveGroup goal that also executes the plan."""
    goal = MoveGroup.Goal()
    goal.request.group_name = group_name
    goal.request.num_planning_attempts = attempts
    goal.request.allowed_planning_time = planning_time
    goal.request.max_velocity_scaling_factor = velocity
    goal.request.max_acceleration_scaling_factor = acceleration
    goal.request.goal_constraints = [constraints]
    goal.planning_options.plan_only = False
    goal.planning_options.replan = False
    return goal


def make_home_trajectory(duration=HOME_DURATION):
    """Build a JTC goal that drives the arm to the Panda home pose."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [float(position) for position in HOME_JOINTS]
    point.time_from_start = Duration(seconds=duration).to_msg()
    goal.trajectory.points = [point]
    return goal


def make_reach_goal(position, quaternion=GRIPPER_DOWN):
    """Move the hand to ``position`` with the gripper pointing down."""
    constraints = Constraints(
        position_constraints=[
            make_position_constraint(END_EFFECTOR_LINK, position)],
        orientation_constraints=[
            make_orientation_constraint(
                END_EFFECTOR_LINK, quaternion=quaternion)],
    )
    return make_move_goal(ARM_GROUP, constraints, attempts=10,
                          planning_time=10.0)


def make_gripper_goal(position):
    """Move both fingers symmetrically to ``position``."""
    constraints = Constraints(
        joint_constraints=make_joint_constraints(
            HAND_JOINTS, [position, position], tolerance=0.005))
    return make_move_goal(HAND_GROUP, constraints, attempts=1,
                          planning_time=2.0, velocity=0.5, acceleration=0.5)


def make_box_object(position, dimensions, frame_id=ARM_BASE_FRAME,
                    object_id=BOX_OBJECT_ID,
                    operation=CollisionObject.ADD):
    """Build a CollisionObject for an axis-aligned box at ``position``."""
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.w = 1.0

    message = CollisionObject()
    message.header.frame_id = frame_id
    message.id = object_id
    message.primitives = [SolidPrimitive(
        type=SolidPrimitive.BOX,
        dimensions=[float(value) for value in dimensions])]
    message.primitive_poses = [pose]
    message.operation = operation
    return message


def make_removed_object(object_id, frame_id=ARM_BASE_FRAME):
    """Build a CollisionObject that removes ``object_id`` from the scene."""
    message = CollisionObject()
    message.header.frame_id = frame_id
    message.id = object_id
    message.operation = CollisionObject.REMOVE
    return message


def make_attach_command(parent_model=ROBOT_MODEL_NAME,
                        parent_link=GAZEBO_HAND_LINK,
                        child_model=BOX_MODEL_NAME,
                        child_link=BOX_LINK_NAME):
    """
    Command understood by the my_robot_grasp_attach Gazebo plugin.

    Gazebo Classic applies position-controlled joints as teleports, so the
    pads transfer no steady load and a friction pinch grasp does not lift the
    cube. The plugin creates a fixed joint between the two links instead,
    which is what makes the cube follow the hand in the simulator. The parent
    link is panda_link7 because Gazebo has no panda_hand link.
    """
    return 'attach %s %s %s %s' % (
        parent_model, parent_link, child_model, child_link)


def make_attached_object(box_object, link_name=END_EFFECTOR_LINK,
                         touch_links=TOUCH_LINKS):
    """Attach ``box_object`` to the gripper for MoveIt and RViz."""
    attached = AttachedCollisionObject()
    attached.link_name = link_name
    attached.object = box_object
    attached.touch_links = list(touch_links)
    return attached


def make_forward_twist(speed=FORWARD_SPEED):
    """Constant forward velocity command."""
    command = Twist()
    command.linear.x = float(speed)
    command.angular.z = 0.0
    return command


def make_stop_twist():
    """All-zero velocity command."""
    return Twist()


def has_all_joints(joint_state, joint_names):
    """Check that every name in ``joint_names`` is present in the state."""
    if joint_state is None:
        return False
    available = set(joint_state.name)
    return all(name in available for name in joint_names)


def odom_xy(odometry):
    """Extract (x, y) from an Odometry message."""
    position = odometry.pose.pose.position
    return (position.x, position.y)


class MobileManipulationNode(Node):
    """Run the drive -> approach -> grasp -> lift sequence."""

    def __init__(self):
        super().__init__('mobile_manipulation_node')
        self.drive_distance = float(
            self.declare_parameter('drive_distance', DRIVE_DISTANCE).value)
        self.forward_speed = float(
            self.declare_parameter('forward_speed', FORWARD_SPEED).value)
        config_path = str(self.declare_parameter(
            'grasp_config', default_grasp_config_path()).value)
        self.grasp = load_config(config_path)
        self.grasp_config_path = config_path
        self.home_first = bool(
            self.declare_parameter('home_first', True).value)

        self.odom = None
        self.ground_truth = None
        self.joint_state = None
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # move_group's PlanningSceneMonitor subscribes to these two topics on
        # its own private node (the /apply_planning_scene service cannot carry
        # attached objects).
        self.collision_object_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)
        self.attached_pub = self.create_publisher(
            AttachedCollisionObject, '/attached_collision_object', 10)
        self.hand_effort_pub = self.create_publisher(
            Float64MultiArray, HAND_EFFORT_TOPIC, 10)
        # The Gazebo attach plugin welds the cube to the hand and answers on
        # /grasp_attach/status, so the node can verify the weld instead of
        # assuming it happened.
        self.attach_pub = self.create_publisher(
            String, GRASP_ATTACH_TOPIC, 10)
        self.attach_status = None
        self.create_subscription(
            String, GRASP_ATTACH_STATUS_TOPIC, self._on_attach_status, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(
            Odometry, GROUND_TRUTH_TOPIC, self._on_ground_truth, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, ARM_TRAJECTORY_ACTION)
        self.cartesian_client = self.create_client(
            GetCartesianPath, CARTESIAN_SERVICE)
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, EXECUTE_ACTION)
        self.switch_client = self.create_client(
            SwitchController, SWITCH_CONTROLLER_SERVICE)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_odom(self, message):
        self.odom = message

    def _on_attach_status(self, message):
        self.attach_status = message.data

    def _on_ground_truth(self, message):
        self.ground_truth = message

    def _on_joint_state(self, message):
        if has_all_joints(message, ARM_JOINTS):
            self.joint_state = message

    def _spin_for(self, seconds):
        """Spin the node for a wall-clock duration."""
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_stop(self, seconds=STOP_PUBLISH_SECONDS):
        """Publish zero velocity repeatedly so the plugin really stops."""
        deadline = time.monotonic() + seconds
        stop = make_stop_twist()
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_pub.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_odom(self, timeout=30.0):
        """Wait until at least one odometry message has arrived."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.odom is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.odom is not None

    def wait_for_ground_truth(self, timeout=GROUND_TRUTH_TIMEOUT):
        """Wait for the first ground-truth base pose from Gazebo."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.ground_truth is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.ground_truth is not None

    def wait_for_arm_joint_states(self, timeout=30.0):
        """Wait for the first complete Panda /joint_states message."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.joint_state is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.joint_state is not None

    def wait_for_move_group(self, timeout=60.0):
        """Wait for MoveIt's /move_action server."""
        return self.move_client.wait_for_server(timeout_sec=timeout)

    def wait_for_base_tf(self, timeout=TF_TIMEOUT):
        """Wait until the base -> odom transform can be looked up."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    ARM_BASE_FRAME, ODOM_FRAME, Time())
                return True
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def base_pose(self):
        """Latest ground-truth base pose, or None if the plugin is absent."""
        if self.ground_truth is None:
            return None
        return self.ground_truth.pose.pose

    def point_in_arm_frame(self, position):
        """
        Transform a world-frame point into the current panda_link0 frame.

        With ground-truth odometry the base pose is exact: rotate the
        world-frame offset into the base frame and add the fixed
        base_link -> panda_link0 translation. Without it (a launch file that
        does not load the plugin) fall back to the odom TF.
        """
        pose = self.base_pose()
        if pose is not None:
            origin = pose.position
            delta = (float(position[0]) - origin.x,
                     float(position[1]) - origin.y,
                     float(position[2]) - origin.z)
            rotation = pose.orientation
            local = rotate_vector(
                inverse_quaternion(
                    (rotation.x, rotation.y, rotation.z, rotation.w)),
                delta)
            # The offset points from base_link to panda_link0, so a point's
            # panda_link0 coordinates are its base coordinates minus it.
            return Point(x=local[0] - ARM_MOUNT_OFFSET[0],
                         y=local[1] - ARM_MOUNT_OFFSET[1],
                         z=local[2] - ARM_MOUNT_OFFSET[2])

        transform = self.tf_buffer.lookup_transform(
            ARM_BASE_FRAME, ODOM_FRAME, Time())
        point = PointStamped()
        point.header.frame_id = ODOM_FRAME
        point.point.x, point.point.y, point.point.z = (
            float(value) for value in position)
        return do_transform_point(point, transform).point

    def orientation_in_arm_frame(self, quaternion=GRIPPER_DOWN):
        """
        Express a world-frame orientation in the current panda_link0 frame.

        The base always stops a few degrees off the heading it started with,
        and panda_link0 turns with it. Constraining the hand to GRIPPER_DOWN
        *in the arm frame* therefore leaves the fingers rotated relative to
        the axis-aligned cube; the pads then hit the cube's edges, torque it
        around and cam it out of the gripper. Rotating the target by the
        measured base yaw keeps the pads parallel to the cube faces.
        """
        pose = self.base_pose()
        if pose is not None:
            rotation = pose.orientation
            return quaternion_multiply(
                inverse_quaternion(
                    (rotation.x, rotation.y, rotation.z, rotation.w)),
                quaternion)

        transform = self.tf_buffer.lookup_transform(
            ARM_BASE_FRAME, ODOM_FRAME, Time())
        rotation = transform.transform.rotation
        return quaternion_multiply(
            (rotation.x, rotation.y, rotation.z, rotation.w), quaternion)

    def publish_collision_objects(self, messages):
        """Add or remove world objects and let MoveIt process the update."""
        for message in messages:
            self.collision_object_pub.publish(message)
        self._spin_for(SCENE_SETTLE_SECONDS)

    def publish_attached_object(self, message):
        """Attach an object to a robot link in MoveIt's planning scene."""
        self.attached_pub.publish(message)
        self._spin_for(SCENE_SETTLE_SECONDS)

    def attach_box_in_gazebo(self, timeout=ATTACH_TIMEOUT):
        """
        Weld the cube to the hand in Gazebo and wait for the plugin's ack.

        The command is repeated until a status arrives: the plugin is loaded
        together with the robot model, so the first messages can be published
        before its subscription exists.
        """
        self.attach_status = None
        command = String(data=make_attach_command())
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.attach_pub.publish(command)
            self._spin_for(ATTACH_REPUBLISH_SECONDS)
            if self.attach_status is None:
                continue
            if self.attach_status == 'attached':
                self.get_logger().info(
                    'Gazebo 已把 %s 固定到 %s 上'
                    % (BOX_MODEL_NAME, END_EFFECTOR_LINK))
                return True
            self.get_logger().error(
                'Gazebo 固定箱子失败：%s' % self.attach_status)
            return False
        self.get_logger().error(
            '等待 Gazebo 固定箱子超时，插件 my_robot_grasp_attach 可能没有加载')
        return False

    def send_home_trajectory(self, timeout=HOME_TIMEOUT):
        """Home the arm through the JTC, bypassing MoveIt collision checks."""
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                '没有找到 %s' % ARM_TRAJECTORY_ACTION)
            return False

        send_future = self.arm_client.send_goal_async(make_home_trajectory())
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送 Home 轨迹超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('机械臂控制器拒绝了 Home 轨迹')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=HOME_DURATION + timeout)
        if not result_future.done():
            self.get_logger().error('等待 Home 轨迹执行结果超时')
            handle.cancel_goal_async()
            return False

        status = result_future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Home 轨迹执行失败，status=%d' % status)
            return False
        return True

    def send_move_goal(self, goal, timeout=MOVE_TIMEOUT):
        """Send a MoveGroup goal and wait for execution to finish."""
        send_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送 MoveGroup 目标超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveIt 拒绝了目标')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error('等待 MoveIt 执行结果超时')
            handle.cancel_goal_async()
            return False

        result = result_future.result()
        error_code = result.result.error_code.val if result else None
        if error_code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                'MoveIt 规划/执行失败，MoveItErrorCodes=%s' % error_code)
            return False
        return True

    def send_cartesian_path(self, waypoints, timeout=MOVE_TIMEOUT,
                            velocity=0.3):
        """Plan a straight-line motion for the hand and execute it."""
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(
                '没有找到 %s' % CARTESIAN_SERVICE)
            return False

        future = self.cartesian_client.call_async(
            make_cartesian_request(waypoints, velocity=velocity))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            self.get_logger().error('计算笛卡尔路径超时')
            return False

        response = future.result()
        if response is None:
            self.get_logger().error('笛卡尔路径服务没有返回结果')
            return False
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                '笛卡尔路径规划失败，MoveItErrorCodes=%s'
                % response.error_code.val)
            return False
        if response.fraction < CARTESIAN_MIN_FRACTION:
            self.get_logger().error(
                '笛卡尔路径只完成 %.1f%%（被碰撞挡住）'
                % (response.fraction * 100.0))
            return False

        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                '没有找到 %s' % EXECUTE_ACTION)
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        send_future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送笛卡尔轨迹超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveIt 拒绝了笛卡尔轨迹')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().error('等待笛卡尔轨迹执行结果超时')
            handle.cancel_goal_async()
            return False

        result = result_future.result()
        if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                '笛卡尔轨迹执行失败，status=%s'
                % (None if result is None else result.status))
            return False
        if result.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                '笛卡尔轨迹执行失败，MoveItErrorCodes=%s'
                % result.result.error_code.val)
            return False
        return True

    def finger_positions(self):
        """Return both finger joint positions, or None if unavailable."""
        if self.joint_state is None:
            return None
        positions = dict(zip(self.joint_state.name,
                             self.joint_state.position))
        if not all(name in positions for name in HAND_JOINTS):
            return None
        return [positions[name] for name in HAND_JOINTS]

    def switch_controllers(self, activate, deactivate, timeout=15.0):
        """Strictly swap which controller owns the finger command interfaces."""
        if not self.switch_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                '没有找到 %s' % SWITCH_CONTROLLER_SERVICE)
            return False
        future = self.switch_client.call_async(
            make_switch_request(activate, deactivate))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error('切换夹爪控制器超时')
            return False
        if not future.result().ok:
            self.get_logger().error('controller_manager 拒绝了夹爪控制器切换')
            return False
        return True

    def squeeze_gripper(self):
        """Press the cube with a constant force and verify the pads touch it."""
        size = self.grasp['size']
        effort = self.grasp['squeeze_effort']
        if not self.switch_controllers(
                activate=[HAND_EFFORT_CONTROLLER],
                deactivate=[HAND_POSITION_CONTROLLER]):
            return False

        command = make_effort_command(effort)
        deadline = time.monotonic() + SQUEEZE_SETTLE_SECONDS
        while rclpy.ok() and time.monotonic() < deadline:
            self.hand_effort_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.02)

        positions = self.finger_positions()
        if positions is None:
            self.get_logger().error('没有收到夹爪关节状态')
            return False
        if not pads_hold_a_cube(positions, size):
            self.get_logger().error(
                '夹爪没有夹住箱子：关节位置 (%0.4f, %0.4f) m，两指间距 '
                '%0.4f m，箱子宽 %0.3f m'
                % (positions[0], positions[1], finger_gap(positions), size))
            return False
        self.get_logger().info(
            '夹爪已用 %.1f N 的力夹住箱子，关节位置 (%0.4f, %0.4f) m。'
            % (effort, positions[0], positions[1]))
        return True

    def drive_forward(self, distance=None):
        """Drive straight until /odom reports ``distance`` metres travelled."""
        if distance is None:
            distance = self.drive_distance
        if self.odom is None:
            self.get_logger().error('没有 /odom，无法开始行驶')
            return False

        start_xy = odom_xy(self.odom)
        deadline = time.monotonic() + DRIVE_TIMEOUT
        self.get_logger().info(
            '开始前进 %.2f m（速度 %.2f m/s）...'
            % (distance, self.forward_speed))
        while rclpy.ok() and time.monotonic() < deadline:
            self.cmd_pub.publish(make_forward_twist(self.forward_speed))
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom is None:
                continue
            travelled = odom_distance(start_xy, odom_xy(self.odom))
            if travelled >= distance:
                self.get_logger().info(
                    '已行驶 %.3f m，停车。' % travelled)
                return True

        self.get_logger().error(
            '行驶 %.1f s 仍未到达 %.2f m，停车。' % (DRIVE_TIMEOUT, distance))
        return False

    def stop_and_settle(self):
        """Send zero velocity, then wait for the base to stop moving."""
        self.get_logger().info('发送零速度并等待底盘停稳 ...')
        self._publish_stop(STOP_PUBLISH_SECONDS)
        self._spin_for(SETTLE_SECONDS)

    def run(self):
        """Execute the complete pick-up-and-lift sequence."""
        self.get_logger().info('等待 /odom ...')
        if not self.wait_for_odom():
            self.get_logger().error('超时：没有收到 /odom')
            return False

        self.get_logger().info('等待 Panda 关节状态 ...')
        if not self.wait_for_arm_joint_states():
            self.get_logger().error('超时：/joint_states 中没有 Panda 关节')
            return False

        self.get_logger().info('等待 MoveIt /move_action ...')
        if not self.wait_for_move_group():
            self.get_logger().error('超时：没有找到 /move_action')
            return False

        if self.home_first:
            self.get_logger().info('第一步：机械臂回 Home 姿态 ...')
            if not self.send_home_trajectory():
                self.get_logger().error('回 Home 失败')
                return False

        self.get_logger().info('第二步：张开夹爪 ...')
        if not self.send_move_goal(make_gripper_goal(GRIPPER_OPEN)):
            self.get_logger().error('张开夹爪失败')
            return False

        self.get_logger().info('第三步：向前行驶 ...')
        if not self.drive_forward():
            self.stop_and_settle()
            return False

        self.stop_and_settle()

        self.get_logger().info('第四步：读取箱子位姿 ...')
        if self.wait_for_ground_truth():
            pose = self.base_pose()
            self.get_logger().info(
                '底盘真实位姿（世界系）：(%0.4f, %0.4f, %0.4f)，'
                'q=(%0.3f, %0.3f, %0.3f, %0.3f)'
                % (pose.position.x, pose.position.y, pose.position.z,
                   pose.orientation.x, pose.orientation.y,
                   pose.orientation.z, pose.orientation.w))
        else:
            self.get_logger().warning(
                '没有收到 %s，退回轮式里程计 TF（抓取可能有厘米级偏差）'
                % GROUND_TRUTH_TOPIC)
            if not self.wait_for_base_tf():
                self.get_logger().error(
                    '超时：没有 %s -> %s 的 TF'
                    % (ODOM_FRAME, ARM_BASE_FRAME))
                return False

        box = self.point_in_arm_frame(box_pose(self.grasp))
        pedestal = self.point_in_arm_frame(pedestal_pose(self.grasp))
        box_xyz = (box.x, box.y, box.z)
        self.get_logger().info(
            '箱子中心在 %s 下为 (%0.3f, %0.3f, %0.3f)。'
            % (ARM_BASE_FRAME, box.x, box.y, box.z))

        size = self.grasp['size']
        box_object = make_box_object(
            box_xyz, (size, size, size), object_id=BOX_OBJECT_ID)
        pedestal_object = make_box_object(
            (pedestal.x, pedestal.y, pedestal.z),
            pedestal_dimensions(self.grasp),
            object_id=PEDESTAL_OBJECT_ID)
        self.publish_collision_objects([box_object, pedestal_object])

        # The base stops yawed relative to the cube; aim the fingers along the
        # cube's faces rather than along the arm frame.
        gripper_orientation = self.orientation_in_arm_frame()
        self.get_logger().info(
            '按底盘实际朝向补偿夹爪姿态：q=(%0.3f, %0.3f, %0.3f, %0.3f)'
            % gripper_orientation)

        tcp_offset = self.grasp['tcp_offset']
        approach_height = size / 2.0 + self.grasp['approach_clearance']
        pre_grasp = tcp_hand_position(box_xyz, tcp_offset, approach_height)
        grasp = tcp_hand_position(box_xyz, tcp_offset)
        lift = tcp_hand_position(
            box_xyz, tcp_offset, self.grasp['lift_height'])

        self.get_logger().info(
            '第五步：伸到箱子上方 %.0f cm，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (self.grasp['approach_clearance'] * 100.0,
               pre_grasp[0], pre_grasp[1], pre_grasp[2], ARM_BASE_FRAME))
        if not self.send_move_goal(
                make_reach_goal(pre_grasp, gripper_orientation)):
            self.get_logger().error('伸向箱子上方失败')
            return False

        self.get_logger().info(
            '第六步：垂直下探到箱子中心，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (grasp[0], grasp[1], grasp[2], ARM_BASE_FRAME))
        if not self.send_cartesian_path(
                [make_pose(grasp, gripper_orientation)]):
            self.get_logger().error('下探到箱子中心失败')
            return False

        pre_close = pre_close_position(size)
        self.get_logger().info(
            '第七步：预闭合夹爪到 %.3f m，再用 %.1f N 的力夹紧 %.2f m 的箱子 ...'
            % (pre_close, self.grasp['squeeze_effort'], size))
        # MoveIt would reject a hand trajectory that drives the pads into the
        # cube while it is still a world object. Remove it from the scene
        # first; the attach message below puts it back as an attached body.
        self.publish_collision_objects([make_removed_object(BOX_OBJECT_ID)])
        if not self.send_move_goal(make_gripper_goal(pre_close)):
            self.get_logger().error('预闭合夹爪失败')
            return False
        if not self.squeeze_gripper():
            self.get_logger().error('夹紧箱子失败')
            return False

        self.get_logger().info(
            '第八步：把箱子挂到 %s 上 ...' % END_EFFECTOR_LINK)
        # Gazebo needs a real fixed joint before the arm moves again: the
        # friction contact alone does not carry the cube.
        if not self.attach_box_in_gazebo():
            self.get_logger().error('抬起箱子失败：箱子没有固定到夹爪上')
            return False
        # MoveIt/RViz carry the cube through the attached collision object.
        self.publish_attached_object(make_attached_object(box_object))

        self.get_logger().info(
            '第九步：垂直向上抬起 %.0f cm，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (self.grasp['lift_height'] * 100.0,
               lift[0], lift[1], lift[2], ARM_BASE_FRAME))
        if not self.send_cartesian_path(
                [make_pose(lift, gripper_orientation)],
                velocity=self.grasp['lift_velocity']):
            self.get_logger().error('抬起箱子失败')
            return False

        self.get_logger().info(
            '抓取完成：箱子已被夹爪提起 %.0f cm。'
            % (self.grasp['lift_height'] * 100.0))
        return True


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = MobileManipulationNode()
    exit_code = 1
    try:
        exit_code = 0 if node.run() else 1
    except KeyboardInterrupt:
        node.get_logger().warning('用户中断')
    finally:
        try:
            node.stop_and_settle()
        except Exception:  # noqa: BLE001 - shutdown must never mask the result
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
