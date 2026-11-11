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
    6. put the grasp cube and the long table into MoveIt's planning scene at
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

The cube pose is configured in config/grasp_box.yaml (the launch file spawns
that cube in Gazebo; the table comes from demo.world) and is expressed in the
Gazebo world frame. The live base pose comes from libgazebo_ros_p3d's ground-truth
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
from controller_manager_msgs.srv import ListControllers, SwitchController
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
    table_dimensions,
    table_pose,
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
# A position goal at the open limit can settle a few millimetres short; the
# wrist camera's side-face bias then lets a descending pad scrape the cube's
# top edge (measured live: the cube was dragged 1.6 cm sideways during the
# descend). The effort controller drives both fingers against their
# mechanical stops, so the open gap is verifiable before the arm moves down.
GRIPPER_OPEN_EFFORT = 10.0  # N per finger
GRIPPER_OPEN_TOLERANCE = 0.001  # m; both fingers must be this close to 0.04
OPEN_TIMEOUT = 2.0  # s

# The gripper is opened and pre-closed by the position JTC, then switched to
# a dedicated effort controller for the squeeze. A position command is a
# teleport in gazebo_ros2_control and generates no steady normal force, and a
# position-derived PID is unstable on the 15 g fingers.
HAND_POSITION_CONTROLLER = 'panda_hand_controller'
HAND_EFFORT_CONTROLLER = 'panda_hand_effort_controller'
HAND_EFFORT_TOPIC = '/panda_hand_effort_controller/commands'
SWITCH_CONTROLLER_SERVICE = '/controller_manager/switch_controller'
LIST_CONTROLLERS_SERVICE = '/controller_manager/list_controllers'
# Clearance between each pad and the cube for the base node's fixed pre-close,
# and the largest lateral cube-to-hand offset the visual grasp will descend
# with. If a pad does touch the cube's top edge on the way down, the contact
# pushes the cube sideways and the force press can end up on its edges; the
# measured failure left the fingers at 7.19 cm, near the cube's 7.07 cm face
# diagonal. The visual grasp therefore measures the top face properly and
# verifies the gripper is fully open before it descends.
PRE_CLOSE_CLEARANCE = 0.012  # m between pad and cube before the force press
# The force press starts from a position pre-close. Its travel is what turns
# the finger's 10 N into an impact instead of a squeeze, so keep the pads a
# few millimetres (not a full centimetre) outside the face that the measured
# lateral offset says is nearest.
PRE_CLOSE_MARGIN = 0.004  # m between the nearest pad and the cube face
# MoveIt plans the gripper goal inside this joint tolerance. It used to be
# 5 mm, and the measured result was pads at 0.0350 m for a 0.031 m goal: the
# error ate the whole PRE_CLOSE_MARGIN and nearly doubled the press travel, so
# the 10 N landed on the 50 g cube as an impact instead of seating on it.
GRIPPER_GOAL_TOLERANCE = 0.001  # m
# A pre-close goal can be reported as executed before the joint states move:
# measured live, both pads still read the open stop 0.4 s after the goal and
# only closed about two seconds later. Read once after a fixed settle and a
# working controller looks dead, so keep polling for this long.
PRE_CLOSE_MOVE_TIMEOUT = 3.0  # s
PRE_CLOSE_POLL = 0.05  # s between finger-position reads while waiting
# The pre-close is a direct JointTrajectoryController goal, not a MoveIt goal.
# MoveIt's hand plan has no useful collision check here (the cube object is
# removed just before closing) and its JTC goal can report "Goal reached"
# while the fingers are still within the controller's 1 cm goal tolerance:
# live at y=+0.6 m the target was 0.0299 m, the open stop 0.0394 m (9.5 mm
# away), and the JTC declared success in 0.27 s without moving a finger.
# Three position reads over 3 s then saw the open stop and the grasp aborted
# with "预闭合没有生效" even though the controller had accepted the goal.
# A direct trajectory with an explicit duration moves the pads and the
# position readback below verifies where they really stopped.
HAND_TRAJECTORY_ACTION = '/panda_hand_controller/follow_joint_trajectory'
PRE_CLOSE_MOVE_DURATION = 0.5  # s for the direct finger trajectory
# Readback error above this is logged: it becomes the length of the force
# press that follows, so it is worth seeing in the log.
PRE_CLOSE_VERIFY_TOLERANCE = 0.002  # m
# How many times to re-send a pre-close that left the pads on the open stop.
PRE_CLOSE_ATTEMPTS = 2
# A pre-close only counts as "did not move" when it asked for this much travel
# away from the open stop.
PRE_CLOSE_MIN_TRAVEL = 0.002  # m
# How long to keep pressing with the effort controller. The fingers can take
# longer than the old fixed 0.8 s when the cube sits slightly off-centre:
# one pad reaches it first and pushes it towards the other. Polling the joint
# states until the gap matches is both faster when it seats early and safe
# when it needs more time.
SQUEEZE_TIMEOUT = 5.0  # s
SQUEEZE_CONTACT_TOLERANCE = 0.003  # m; pads must rest on the cube faces
# The pads sink into the cube before ODE balances the 10 N press: every
# measured seat left the pad gap at 0.0440-0.0443 m for the 0.05 m cube, i.e.
# exactly the old lower bound. Allow the measured penetration on the low side
# so a slightly deeper seat does not discard a valid grasp.
SQUEEZE_SEAT_PENETRATION = 0.003  # m of extra gap loss below the cube width
# How many presses to try when one never moved the pads off the open stop.
# The retry re-activates the effort controller first.
SQUEEZE_ATTEMPTS = 2
# How long to let the pads settle after the weld releases the squeeze. The
# effort controller keeps its last command, so without this the 10 N press
# keeps driving the pad into the now rigidly attached cube.
GRIP_RELEASE_SETTLE = 0.2  # s

DRIVE_DISTANCE = 1.0
FORWARD_SPEED = 0.25
STOP_PUBLISH_SECONDS = 1.0
SETTLE_SECONDS = 0.5
DRIVE_TIMEOUT = 30.0
MOVE_TIMEOUT = 90.0
TF_TIMEOUT = 15.0
SCENE_SETTLE_SECONDS = 0.5

# MoveIt occasionally returns an invalid plan (-2) from the free-space
# approach even though the pose is reachable: at the last validation step one
# interpolated state is found in collision and the whole request fails. A
# fresh planning attempt uses a new random seed and normally succeeds, so
# retry the retryable planning errors instead of aborting the grasp.
MOVE_ATTEMPTS = 3
RETRYABLE_MOVE_ERRORS = (
    MoveItErrorCodes.PLANNING_FAILED,
    MoveItErrorCodes.INVALID_MOTION_PLAN,
)


def move_should_retry(error_code):
    """Return True when a failed MoveGroup result is worth replanning."""
    return error_code in RETRYABLE_MOVE_ERRORS


POSITION_TOLERANCE = 0.01  # m
ORIENTATION_TOLERANCE = 0.05  # rad, about 2.9 degrees
JOINT_TOLERANCE = 0.01  # rad or m

BOX_OBJECT_ID = 'grasp_box'
TABLE_OBJECT_ID = 'grasp_table'

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


def pre_close_position_for(size, lateral, margin=PRE_CLOSE_MARGIN):
    """
    Finger value that leaves ``margin`` between the nearest pad and the cube.

    ``lateral`` is how far the cube centre is from the hand centre along the
    finger axis. With the cube pushed towards one pad by that much, the
    nearest pad's distance to the cube face is
    ``size / 2 + lateral - finger_value``; solving for the requested margin
    gives the value below. The far pad is then ``2 * lateral`` further away,
    so a small measured offset keeps both pads close to their faces and the
    effort press has little room to accelerate.
    """
    offset = max(0.0, float(lateral))
    value = float(size) / 2.0 + offset + float(margin)
    return min(GRIPPER_OPEN, value)


def finger_gap(positions):
    """
    Gap between the pads for the two finger joint values.

    Each finger joint value is one pad's displacement from the closed pose,
    so the pads' inner faces are ``sum(positions)`` apart.
    """
    return float(positions[0]) + float(positions[1])


def gripper_stuck_open(positions, target,
                       tolerance=GRIPPER_OPEN_TOLERANCE,
                       min_travel=PRE_CLOSE_MIN_TRAVEL):
    """
    Return True when a closing command left both pads on the open stop.

    ``target`` is the finger value the command asked for. A goal that is
    planned but never applied leaves the pads at 0.04 m (measured: a whole
    squeeze reported (0.0400, 0.0400) m), so this separates a dead command
    path from a cube that actually stopped the fingers. Targets within
    ``min_travel`` of the stop are not judged: the pads have not been asked
    to move far enough for the difference to mean anything.
    """
    return (min(positions) >= GRIPPER_OPEN - float(tolerance)
            and float(target) <= GRIPPER_OPEN - float(min_travel))


def wait_for_pad_motion(read_positions, target,
                        timeout=PRE_CLOSE_MOVE_TIMEOUT,
                        poll=PRE_CLOSE_POLL,
                        clock=time.monotonic,
                        sleep=time.sleep):
    """
    Read the finger positions until the pads leave the open stop.

    The pre-close goal and the joint states are not synchronised: a goal
    reported as executed can still show both pads on the open stop for a
    moment. Waiting for the pads to move costs nothing when the controller is
    prompt (the first read already shows motion) and turns the measured
    late-execution case into a success instead of a false failure.
    """
    positions = read_positions()
    deadline = clock() + timeout
    while positions is not None and gripper_stuck_open(positions, target) \
            and clock() < deadline:
        sleep(poll)
        positions = read_positions()
    return positions


def squeeze_seat_window(size, tolerance=SQUEEZE_CONTACT_TOLERANCE,
                        penetration=SQUEEZE_SEAT_PENETRATION):
    """
    Pad-gap range that counts as resting on two opposite cube faces.

    The high side is the face-grip tolerance for a cube that sits slightly
    off-centre. The low side also allows the measured penetration of the pads
    into the cube under the 10 N press.
    """
    half = float(tolerance)
    return (float(size) - 2.0 * (half + float(penetration)),
            float(size) + 2.0 * half)


def pads_hold_a_cube(positions, size, tolerance=SQUEEZE_CONTACT_TOLERANCE,
                     penetration=SQUEEZE_SEAT_PENETRATION):
    """
    Return True when the pads closed on the cube instead of past it.

    The cube can sit a little off-centre between the pads: whichever finger
    touches first pushes the cube towards the other one, and friction stops it
    from sliding back. What must match the cube width is therefore the pad
    gap, not each finger separately. A missed grasp leaves the fingers fully
    closed and the gap near zero.
    """
    low, high = squeeze_seat_window(size, tolerance, penetration)
    return low <= finger_gap(positions) <= high


def cube_diagonal(size):
    """Widest planar span of the cube: the diagonal of a square face."""
    return float(size) * math.sqrt(2.0)


def squeeze_outcome(positions, size, tolerance=SQUEEZE_CONTACT_TOLERANCE,
                    penetration=SQUEEZE_SEAT_PENETRATION):
    """
    Classify why the squeeze stopped where it did.

    ``held`` means the pads rest on opposite faces. ``corners`` is the
    measured failure mode of a rotated cube: both pads press opposite
    vertical edges, so the gap is the face diagonal, sqrt(2) * size.
    ``missed`` means the fingers closed completely without the cube between
    them; ``blocked`` is anything else (the cube on the table, an arm link,
    or a stuck finger).
    """
    gap = finger_gap(positions)
    if pads_hold_a_cube(positions, size, tolerance, penetration):
        return 'held'
    if abs(gap - cube_diagonal(size)) <= 2.0 * tolerance:
        return 'corners'
    if gap <= float(size) / 2.0:
        return 'missed'
    return 'blocked'


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


def make_open_effort_command(effort, joint_count=len(HAND_JOINTS)):
    """Build the command that pushes both fingers against their open stops."""
    message = Float64MultiArray()
    message.data = [abs(float(effort))] * joint_count
    return message


def make_release_command(joint_count=len(HAND_JOINTS)):
    """
    Build the zero-effort command that stops the squeeze after the weld.

    The effort controller holds its last command indefinitely. Once the cube
    is welded to the hand it can no longer yield, so keeping 10 N on the pads
    over-constrains the contact: the stiff pad bounces out again and the finger
    chatters between the pressed and the free pose at the control rate
    (measured in Gazebo: 0.0179 <-> 0.0240 m, i.e. 6 mm). The wrist camera
    aliases that to a white pad jumping left and right at the cube's edge.
    Zero effort leaves the pads resting on the cube without the chatter.
    """
    message = Float64MultiArray()
    message.data = [0.0] * joint_count
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


def quaternion_yaw(quaternion):
    """Yaw of a (x, y, z, w) quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


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


def point_in_world(local, base_pose, offset=ARM_MOUNT_OFFSET):
    """
    Map a panda_link0 point into the world frame for a base pose.

    ``base_pose`` is the ground-truth pose of base_link: ``local`` gains the
    fixed base_link -> panda_link0 offset, is rotated by the base yaw and is
    then translated by the base position. The inverse is ``point_in_arm``.
    """
    translation = base_pose.position
    rotation = base_pose.orientation
    lifted = (float(local[0]) + offset[0],
              float(local[1]) + offset[1],
              float(local[2]) + offset[2])
    delta = rotate_vector(
        (rotation.x, rotation.y, rotation.z, rotation.w), lifted)
    return (translation.x + delta[0],
            translation.y + delta[1],
            translation.z + delta[2])


def point_in_arm(point, base_pose, offset=ARM_MOUNT_OFFSET):
    """
    Map a world-frame point into panda_link0 for a base pose.

    This is the exact inverse of ``point_in_world``. It is used to keep a
    vision measurement valid after the unbraked chassis has moved: the cube
    is static in the world, so the world point stays fixed while its
    panda_link0 coordinates are recomputed from the current base pose.
    """
    translation = base_pose.position
    rotation = base_pose.orientation
    delta = (float(point[0]) - translation.x,
             float(point[1]) - translation.y,
             float(point[2]) - translation.z)
    local = rotate_vector(
        inverse_quaternion(
            (rotation.x, rotation.y, rotation.z, rotation.w)), delta)
    return (local[0] - offset[0],
            local[1] - offset[1],
            local[2] - offset[2])


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


def make_gripper_trajectory(position, duration=PRE_CLOSE_MOVE_DURATION):
    """Direct JTC goal that moves both fingers to ``position``."""
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(HAND_JOINTS)
    point = JointTrajectoryPoint()
    point.positions = [float(position), float(position)]
    point.time_from_start = Duration(seconds=float(duration)).to_msg()
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
            HAND_JOINTS, [position, position],
            tolerance=GRIPPER_GOAL_TOLERANCE))
    return make_move_goal(HAND_GROUP, constraints, attempts=1,
                          planning_time=2.0, velocity=0.5, acceleration=0.5)


def make_box_object(position, dimensions, frame_id=ARM_BASE_FRAME,
                    object_id=BOX_OBJECT_ID,
                    operation=CollisionObject.ADD, yaw=0.0):
    """Build a CollisionObject for a box at ``position``, rotated by ``yaw``."""
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    # The long table must keep its world orientation when the base faces the
    # cube at an angle; the cube itself is square, so yaw is irrelevant there.
    pose.orientation.z = math.sin(float(yaw) / 2.0)
    pose.orientation.w = math.cos(float(yaw) / 2.0)

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

    def __init__(self, node_name='mobile_manipulation_node'):
        super().__init__(node_name)
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
        self.hand_client = ActionClient(
            self, FollowJointTrajectory, HAND_TRAJECTORY_ACTION)
        self.cartesian_client = self.create_client(
            GetCartesianPath, CARTESIAN_SERVICE)
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, EXECUTE_ACTION)
        self.switch_client = self.create_client(
            SwitchController, SWITCH_CONTROLLER_SERVICE)
        self.list_client = self.create_client(
            ListControllers, LIST_CONTROLLERS_SERVICE)
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
            x, y, z = point_in_arm(position, pose)
            return Point(x=x, y=y, z=z)

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

    def table_yaw_in_arm_frame(self):
        """
        Yaw that keeps the world-aligned table aligned in panda_link0.

        The table model in demo.world is axis-aligned. When the base stops
        facing the cube at an angle, the table appears rotated by the
        opposite yaw in the arm frame; publishing an unrotated long box would
        put its far half in the wrong place in the planning scene.
        """
        pose = self.base_pose()
        if pose is not None:
            rotation = pose.orientation
            return -quaternion_yaw(
                (rotation.x, rotation.y, rotation.z, rotation.w))

        # No ground truth: odom is world-aligned at the start of a run, so
        # the arm <- odom yaw is already the table's yaw in the arm frame.
        transform = self.tf_buffer.lookup_transform(
            ARM_BASE_FRAME, ODOM_FRAME, Time())
        rotation = transform.transform.rotation
        return quaternion_yaw(
            (rotation.x, rotation.y, rotation.z, rotation.w))

    def make_table_object(self):
        """Collision object for the fixed long table, in panda_link0."""
        centre = self.point_in_arm_frame(table_pose(self.grasp))
        return make_box_object(
            (centre.x, centre.y, centre.z), table_dimensions(self.grasp),
            object_id=TABLE_OBJECT_ID,
            yaw=self.table_yaw_in_arm_frame())

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

    def send_hand_trajectory(self, position, timeout=MOVE_TIMEOUT):
        """
        Move both fingers through the position JTC and wait for the result.

        Unlike a MoveIt goal, the trajectory carries an explicit duration, so
        the controller starts moving immediately. The caller still checks the
        joint readback: the action result says the pads reached the commanded
        value, not where the grasp left them.
        """
        if not self.hand_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                '没有找到 %s' % HAND_TRAJECTORY_ACTION)
            return False

        send_future = self.hand_client.send_goal_async(
            make_gripper_trajectory(position))
        rclpy.spin_until_future_complete(
            self, send_future, timeout_sec=timeout)
        if not send_future.done():
            self.get_logger().error('发送夹爪轨迹超时')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('夹爪控制器拒绝了预闭合轨迹')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future,
            timeout_sec=PRE_CLOSE_MOVE_DURATION + timeout)
        if not result_future.done():
            self.get_logger().error('等待夹爪轨迹执行结果超时')
            handle.cancel_goal_async()
            return False

        result = result_future.result()
        status = result.status
        if status != GoalStatus.STATUS_SUCCEEDED:
            # The JTC aborts (status=6) when its goal/path tolerance is
            # violated; error_code tells which. 0.0321 m against a 0.0299 m
            # target is 2.2 mm of real seating error, not a dead command.
            error_code = getattr(result.result, 'error_code', None)
            error_string = getattr(result.result, 'error_string', None)
            self.get_logger().error(
                '预闭合轨迹执行失败，status=%d，error_code=%s，%s'
                % (status, error_code,
                   error_string or '控制器没有给出原因'))
            return False
        return True

    def send_move_goal(self, goal, timeout=MOVE_TIMEOUT):
        """Send a MoveGroup goal and wait for execution to finish."""
        for attempt in range(1, MOVE_ATTEMPTS + 1):
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
            if error_code == MoveItErrorCodes.SUCCESS:
                return True
            if not move_should_retry(error_code) \
                    or attempt == MOVE_ATTEMPTS:
                self.get_logger().error(
                    'MoveIt 规划/执行失败，MoveItErrorCodes=%s'
                    % error_code)
                return False
            self.get_logger().warning(
                'MoveIt 返回可重试的错误 %s，重新规划（第 %d/%d 次）'
                % (error_code, attempt + 1, MOVE_ATTEMPTS))
            self._spin_for(0.2)
        return False

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

    def read_finger_positions_once(self, poll=PRE_CLOSE_POLL):
        """Spin once so a fresh joint state arrives, then read the fingers."""
        rclpy.spin_once(self, timeout_sec=poll)
        return self.finger_positions()

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

    def controller_states(self, timeout=15.0):
        """Return controller_manager's ``{name: state}``, or None."""
        if not self.list_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                '没有找到 %s' % LIST_CONTROLLERS_SERVICE)
            return None
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error('查询夹爪控制器状态超时')
            return None
        return {controller.name: controller.state
                for controller in future.result().controller}

    def ensure_hand_position_controller(self, timeout=15.0):
        """
        Make the position JTC own the fingers before a pre-close.

        A failed squeeze leaves the effort controller active, and
        ensure_gripper_open() returns early when the pads already look open
        without switching back. A trajectory sent to an inactive JTC can then
        be accepted and reported successful while no finger moves -- the
        "预闭合没有生效" failure. Query the real controller state instead of
        assuming the launch-time owner is still active.
        """
        states = self.controller_states(timeout)
        if states is None:
            return False
        if HAND_POSITION_CONTROLLER not in states:
            self.get_logger().error(
                '%s 没有加载，无法执行位置预闭合' % HAND_POSITION_CONTROLLER)
            return False
        position_active = states[HAND_POSITION_CONTROLLER] == 'active'
        effort_active = states.get(HAND_EFFORT_CONTROLLER) == 'active'
        if position_active and not effort_active:
            return True
        self.get_logger().info(
            '预闭合前恢复位置控制：%s=%s，%s=%s，切换到 %s ...'
            % (HAND_POSITION_CONTROLLER, states[HAND_POSITION_CONTROLLER],
               HAND_EFFORT_CONTROLLER,
               states.get(HAND_EFFORT_CONTROLLER, '未加载'),
               HAND_POSITION_CONTROLLER))
        return self.switch_controllers(
            activate=[] if position_active else [HAND_POSITION_CONTROLLER],
            deactivate=[HAND_EFFORT_CONTROLLER] if effort_active else [],
            timeout=timeout)

    def ensure_gripper_open(self, stage):
        """
        Open both fingers to the mechanical stop and verify the gap.

        The position controller can settle short of the 0.04 m limit. A pad
        that is even a few millimetres less open is enough to catch the cube's
        top edge during the descend, so push against the stops with the effort
        controller and check the joint values before trusting the opening.
        """
        positions = self.finger_positions()
        if positions is not None and min(positions) \
                >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE:
            self.get_logger().info(
                '%s：夹爪已完全张开 (%0.4f, %0.4f) m。'
                % (stage, positions[0], positions[1]))
            return True

        self.get_logger().warning(
            '%s：夹爪开口不足（%s），用 %.0f N 的力顶到机械限位 ...'
            % (stage,
               '未知' if positions is None
               else '(%0.4f, %0.4f) m' % (positions[0], positions[1]),
               GRIPPER_OPEN_EFFORT))
        if not self.switch_controllers(
                activate=[HAND_EFFORT_CONTROLLER],
                deactivate=[HAND_POSITION_CONTROLLER]):
            return False

        command = make_open_effort_command(GRIPPER_OPEN_EFFORT)
        deadline = time.monotonic() + OPEN_TIMEOUT
        while rclpy.ok() and time.monotonic() < deadline:
            self.hand_effort_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.02)
            positions = self.finger_positions()
            if positions is not None and min(positions) \
                    >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE:
                break

        opened = positions is not None and min(positions) \
            >= GRIPPER_OPEN - GRIPPER_OPEN_TOLERANCE
        # The pre-close is a position goal, so the JTC has to own the
        # interfaces again before the gripper is used for a grasp.
        switched_back = self.switch_controllers(
            activate=[HAND_POSITION_CONTROLLER],
            deactivate=[HAND_EFFORT_CONTROLLER])
        if not opened:
            self.get_logger().error(
                '%s：%.1f s 内夹爪没有顶到最大开口（当前 %s）。'
                % (stage, OPEN_TIMEOUT,
                   '未知' if positions is None
                   else '(%0.4f, %0.4f) m' % (positions[0], positions[1])))
            return False
        if not switched_back:
            return False
        self.get_logger().info(
            '%s：夹爪已顶到最大开口 (%0.4f, %0.4f) m。'
            % (stage, positions[0], positions[1]))
        return True

    def pre_close_gripper(self, position):
        """
        Move the pads to ``position`` and verify where they really stopped.

        The joint states decide, not the goal result: the hand JTC tolerates
        a per-joint goal error (5 mm), so a reported success can still leave
        the pads millimetres from the requested value. That gap is press
        travel (and impact speed) for the force squeeze that follows, so it
        is measured and logged, and a goal that never moved the pads off the
        open stop is sent once more.

        The position JTC must own the finger interfaces first: if the effort
        controller is still active, its goals cannot move the fingers and the
        pre-close silently does nothing.
        """
        if not self.ensure_hand_position_controller():
            self.get_logger().error(
                '预闭合前无法恢复 %s，放弃夹紧以免对着张开的夹爪空按。'
                % HAND_POSITION_CONTROLLER)
            return False
        positions = None
        for attempt in range(1, PRE_CLOSE_ATTEMPTS + 1):
            if not self.send_hand_trajectory(position):
                # The JTC aborts when the pads miss the goal by more than its
                # tolerance. Report where they actually stopped: this is the
                # difference between "the command never reached the fingers"
                # and "the pads seated a couple of millimetres short of the
                # requested pre-close", which the force squeeze can close.
                positions = self.finger_positions()
                if positions is not None:
                    error = max(
                        abs(value - position) for value in positions)
                    self.get_logger().error(
                        '预闭合失败时两指停在 (%0.4f, %0.4f) m，目标 %.4f m，'
                        '最大偏差 %.1f mm。'
                        % (positions[0], positions[1], position,
                           error * 1000.0))
                return False
            positions = wait_for_pad_motion(
                self.read_finger_positions_once, position)
            if positions is None:
                self.get_logger().warning(
                    '预闭合后没有收到夹爪关节状态，无法核对两指位置。')
                return True
            if not gripper_stuck_open(positions, position):
                error = max(abs(value - position) for value in positions)
                if error > PRE_CLOSE_VERIFY_TOLERANCE:
                    self.get_logger().warning(
                        '预闭合后两指停在 (%0.4f, %0.4f) m，与目标 %.4f m '
                        '相差 %.1f mm；这段距离会由随后的夹紧补上。'
                        % (positions[0], positions[1], position,
                           error * 1000.0))
                else:
                    self.get_logger().info(
                        '预闭合后两指停在 (%0.4f, %0.4f) m，与目标一致。'
                        % (positions[0], positions[1]))
                return True
            self.get_logger().warning(
                '预闭合没有生效：两指仍停在张开限位 (%0.4f, %0.4f) m，'
                '目标 %.4f m。重新发送预闭合目标（第 %d/%d 次）...'
                % (positions[0], positions[1], position,
                   attempt + 1, PRE_CLOSE_ATTEMPTS))
        self.get_logger().error(
            '预闭合失败：两指仍停在张开限位 (%0.4f, %0.4f) m，目标 %.4f m。'
            '夹爪没有执行位置指令，不再继续夹紧，以免对着张开的夹爪空按。'
            % (positions[0], positions[1], position))
        return False

    def press_until_seated(self, command, size, timeout=SQUEEZE_TIMEOUT):
        """
        Publish ``command`` until the pads hold the cube or ``timeout`` ends.

        Returns the last finger positions seen (None when no joint states
        arrived). The caller checks them: after a press that ends on the open
        stop the pads never moved at all, which is a different failure from a
        cube that stopped them part way.
        """
        positions = None
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            self.hand_effort_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.02)
            positions = self.finger_positions()
            if positions is not None and pads_hold_a_cube(positions, size):
                break
        return positions

    def restart_effort_controller(self):
        """
        Re-activate the effort controller so it re-applies its command.

        The effort controller is open loop and keeps its last command while
        deactivated, so a fresh activation is a cheap way to recover the
        command path when a press did not move the fingers at all.
        """
        if not self.switch_controllers(
                activate=[HAND_POSITION_CONTROLLER],
                deactivate=[HAND_EFFORT_CONTROLLER]):
            return False
        return self.switch_controllers(
            activate=[HAND_EFFORT_CONTROLLER],
            deactivate=[HAND_POSITION_CONTROLLER])

    def squeeze_gripper(self):
        """Press the cube with a constant force and verify the pads touch it."""
        size = self.grasp['size']
        effort = self.grasp['squeeze_effort']
        command = make_effort_command(effort)
        positions = None
        for attempt in range(1, SQUEEZE_ATTEMPTS + 1):
            if attempt == 1:
                switched = self.switch_controllers(
                    activate=[HAND_EFFORT_CONTROLLER],
                    deactivate=[HAND_POSITION_CONTROLLER])
            else:
                switched = self.restart_effort_controller()
            if not switched:
                return False

            positions = self.press_until_seated(command, size)
            if positions is None:
                self.get_logger().error('没有收到夹爪关节状态')
                return False
            if pads_hold_a_cube(positions, size):
                self.get_logger().info(
                    '夹爪已用 %.1f N 的力夹住箱子，关节位置 (%0.4f, %0.4f) m。'
                    % (effort, positions[0], positions[1]))
                return True
            if not gripper_stuck_open(positions, size / 2.0):
                break
            # Nothing between the pads can hold them at 0.08 m: the command
            # never reached the fingers. Try once more on a fresh activation
            # instead of blaming the cube.
            self.get_logger().warning(
                '夹紧指令没有生效：%.1f s 内两指一直停在张开限位 '
                '(%0.4f, %0.4f) m。重新激活 %s 后再夹一次（第 %d/%d 次）...'
                % (SQUEEZE_TIMEOUT, positions[0], positions[1],
                   HAND_EFFORT_CONTROLLER, attempt + 1, SQUEEZE_ATTEMPTS))

        if positions is None:
            self.get_logger().error('没有收到夹爪关节状态')
            return False
        if gripper_stuck_open(positions, size / 2.0):
            detail = (
                '两指一直停在张开限位，说明夹爪没有收到闭合力，或者控制器'
                '没有把力加到手指上。这不是方块挡住手指：完全张开时两指'
                '内侧面之间有 0.08 m，而方块只有 %.3f m 宽。请检查 %s 的'
                '切换和 controller_manager 日志'
                % (size, HAND_EFFORT_CONTROLLER))
        else:
            outcome = squeeze_outcome(positions, size)
            if outcome == 'corners':
                detail = (
                    '两指间距接近方块的对角线（%.4f m）。这说明两指没有压在'
                    '两个对面的平面上：要么方块被碰转了约 45 度，要么方块'
                    '整体偏到夹爪一侧、一根手指压在方块的棱上。下探前必须'
                    '把方块和夹爪的中心对齐，并确认夹爪完全张开'
                    % cube_diagonal(size))
            elif outcome == 'missed':
                detail = '两指已完全闭合，方块不在两指之间（下探位置偏了）'
            elif outcome == 'blocked':
                detail = '两指被两指之间的物体挡住，间距既不是方块宽也不是对角线'
            else:
                detail = '夹爪没有进入夹持状态'
        self.get_logger().error(
            '%.1f s 内夹爪没有夹住箱子：关节位置 (%0.4f, %0.4f) m，'
            '两指间距 %0.4f m，箱子宽 %0.3f m。%s'
            % (SQUEEZE_TIMEOUT, positions[0], positions[1],
               finger_gap(positions), size, detail))
        return False

    def release_grip(self, settle=GRIP_RELEASE_SETTLE):
        """
        Stop squeezing once the cube is welded to the hand.

        The weld, not friction, carries the cube from here on, so the press
        only over-constrains the finger against a rigid body. Zero effort lets
        the pad rest on the cube, which stops the 6 mm chatter the wrist
        camera shows and keeps the lift quiet.
        """
        command = make_release_command()
        deadline = time.monotonic() + settle
        while rclpy.ok() and time.monotonic() < deadline:
            self.hand_effort_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().info(
            '方块已由焊点固定，松开夹紧力；夹爪不再对刚性方块施力。')

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
        if not self.ensure_gripper_open('张开夹爪后'):
            self.get_logger().error('夹爪没有完全张开')
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
        box_xyz = (box.x, box.y, box.z)
        self.get_logger().info(
            '箱子中心在 %s 下为 (%0.3f, %0.3f, %0.3f)。'
            % (ARM_BASE_FRAME, box.x, box.y, box.z))

        size = self.grasp['size']
        box_object = make_box_object(
            box_xyz, (size, size, size), object_id=BOX_OBJECT_ID)
        table_object = self.make_table_object()
        self.publish_collision_objects([box_object, table_object])

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
        if not self.pre_close_gripper(pre_close):
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
        # The weld now carries the cube: stop pressing before the lift, or
        # the effort controller chatters the pads against the rigid cube.
        self.release_grip()

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
