"""
Vision-guided grasp: find the red cube and pick it up where it really is.

Pipeline:
    1. wait for /odom, Panda joint states and MoveIt's /move_action;
    2. home the arm, open the gripper and drive to the configured viewpoint;
    3. move the hand to a look pose above the pickup area, so the wrist
       camera sees the cube;
    4. threshold /camera/image_raw in HSV for red and take the centroid of
       the largest blob;
    5. take the depth under that pixel from /camera/depth/image_raw; if the
       depth camera is missing or the patch is invalid, intersect the pixel
       ray with the known height of the cube's top face instead;
    6. back-project with the intrinsics from /camera/camera_info and
       transform the point from camera_optical_link to panda_link0 with TF;
    7. move above the detected point, re-detect for a closer look, descend,
       close, weld and lift.

The arm motions, controller swaps and Gazebo weld are inherited from
mobile_manipulation_node. Vision replaces the cube pose from
config/grasp_box.yaml as the grasp target; that file is still used for the
cube size, the TCP offset and the height of the top face used by the
depth-less fallback.
"""

import math
import time
from collections import namedtuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException

from my_robot_sim.mobile_manipulation_node import (
    ARM_BASE_FRAME,
    BOX_OBJECT_ID,
    GRIPPER_DOWN,
    GRIPPER_OPEN,
    ODOM_FRAME,
    PRE_CLOSE_CLEARANCE,
    TF_TIMEOUT,
    MobileManipulationNode,
    make_attached_object,
    make_box_object,
    make_gripper_goal,
    make_pose,
    make_reach_goal,
    make_removed_object,
    point_in_world,
    pre_close_position_for,
    rotate_vector,
)

IMAGE_TOPIC = '/camera/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera_info'
DEPTH_TOPIC = '/camera/depth/image_raw'
DEBUG_IMAGE_TOPIC = '/visual_grasp/debug_image'

# Red wraps around the hue circle, so two ranges are needed. The thresholds
# are deliberately loose: the cubes are matte and the reward is finding the
# blob, while size and shape filtering are done by contour area.
RED_RANGES = (
    ((0, 100, 60), (10, 255, 255)),
    ((170, 100, 60), (180, 255, 255)),
)
MORPHOLOGY_KERNEL = 5
MIN_BLOB_AREA = 80  # px^2; a 5 cm cube at the look pose covers thousands

# Median over a small patch, instead of the single pixel, survives the depth
# image's mixed pixels along the cube's edges and its invalid zeros.
DEPTH_PATCH_RADIUS = 4

# The cube is seen from the side as well as from above: the colour blob
# contains the top face and one or two side faces. Sampling the depth under
# the blob's centroid therefore returned a point on a side face, up to 2.6 cm
# below the real top and a centimetre off in x/y. The top face is the highest
# set of points in panda_link0, so select the points within this height of the
# highest one and average them: the face is horizontal, so its mean is its
# true centre and its z is the surface the hand must be built from.
TOP_FACE_HEIGHT_TOLERANCE = 0.006  # m, in panda_link0
TOP_FACE_STRIDE = 2  # sample every second pixel of the mask
TOP_FACE_MIN_POINTS = 40

# The wrist camera is mounted over one finger pad, and at the look pose that
# pad hides the near part of the top face: only a strip along the far edge is
# visible and its mean is pulled towards the camera (17 mm against Gazebo
# ground truth, measured on the live camera). That bias placed the hand off
# the cube, so the descend scraped a pad and the cube was never seated. When
# the visible strip is clearly shorter than the cube edge, the far edge is
# still a real cube edge, so the centre is rebuilt half an edge inwards from
# it. A face that is (almost) fully visible, including a cube yawed to 10-45
# degrees, keeps the plain mean, which has no such bias.
FULL_FACE_FRACTION = 0.9

DETECTION_TIMEOUT = 10.0
REFINE_TIMEOUT = 4.0
CAMERA_TIMEOUT = 10.0
DEPTH_TIMEOUT = 5.0

# Let the unbraked chassis stop rocking before sampling the base orientation.
SETTLE_AFTER_MOVE = 0.3  # s

# Hand pose (panda_link0) used to look at the pickup area. The drives use
# wheel odometry, which over-reports the distance by about 15 cm on this
# floor, so after the 1 m drive the cube sits ~0.60 m in front of the arm
# rather than at 0.45 m. Looking from 0.58 m keeps it near the image centre;
# the 60 degree field of view still contains it if the robot stops a few
# centimetres early or late. With the gripper pointing down the camera is
# ~0.42 m above the cube's plane.
LOOK_POSITION = (0.58, 0.0, 0.42)

# A closer re-detection is only trusted if the cube has not jumped in the
# *world* frame. Comparing panda_link0 points would reject a valid refinement
# whenever the unbraked chassis crept while the arm moved, which is exactly
# the case the refinement has to correct. A world shift beyond this is more
# likely an arm part than the cube.
MAX_REFINEMENT_SHIFT = 0.05  # m, between the two vision measurements

Detection = namedtuple('Detection', ['u', 'v', 'area'])

SOURCE_NAMES = {
    'top': '顶面点云',
    'depth': '深度图',
    'plane': '已知高度平面',
}


def source_name(source):
    """Human-readable name of a target_point_in_arm_frame source."""
    return SOURCE_NAMES.get(source, str(source))


def bgr_image(message):
    """Return the sensor_msgs/Image frame as a BGR numpy array."""
    if message.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('unsupported RGB encoding %r' % message.encoding)
    channels = 3
    row_bytes = message.step
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, row_bytes)
    pixels = rows[:, :message.width * channels].reshape(
        message.height, message.width, channels)
    if message.encoding == 'rgb8':
        return pixels[:, :, ::-1].copy()
    return pixels.copy()


def depth_image(message):
    """Return the depth frame in metres as a float32 numpy array."""
    if message.encoding == '32FC1':
        dtype = np.dtype(np.float32)
        scale = 1.0
    elif message.encoding == '16UC1':
        dtype = np.dtype(np.uint16)
        scale = 0.001
    else:
        raise ValueError('unsupported depth encoding %r' % message.encoding)
    row_items = message.step // dtype.itemsize
    rows = np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, row_items)
    pixels = rows[:, :message.width]
    return pixels.astype(np.float32) * np.float32(scale)


def red_mask(image_bgr, ranges=RED_RANGES):
    """Binary mask of the pixels whose HSV colour is red."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask, cv2.inRange(hsv, np.array(lower, np.uint8),
                              np.array(upper, np.uint8)))
    kernel = np.ones((MORPHOLOGY_KERNEL, MORPHOLOGY_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_red_blob(image_bgr, min_area=MIN_BLOB_AREA):
    """
    Return the largest red blob as (u, v, area), or None.

    The centroid comes from the contour moments, so it is the centre of the
    visible red area rather than the centre of its bounding box.
    """
    contours, _ = cv2.findContours(
        red_mask(image_bgr), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments['m00'] <= 0.0:
            continue
        candidate = Detection(
            moments['m10'] / moments['m00'],
            moments['m01'] / moments['m00'],
            area)
        if best is None or candidate.area > best.area:
            best = candidate
    return best


def camera_intrinsics(info):
    """Return (fx, fy, cx, cy) from a CameraInfo message."""
    k = info.k
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def scale_pixel(u, v, source_shape, target_shape):
    """Map a pixel from one image size to another (RGB vs depth)."""
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    return (u * target_width / float(source_width),
            v * target_height / float(source_height))


def depth_from_patch(depth, u, v, radius=DEPTH_PATCH_RADIUS):
    """Median of the valid depths around (u, v), or None if all invalid."""
    height, width = depth.shape[:2]
    centre_u = int(round(u))
    centre_v = int(round(v))
    left = max(0, centre_u - radius)
    right = min(width, centre_u + radius + 1)
    top = max(0, centre_v - radius)
    bottom = min(height, centre_v + radius + 1)
    patch = depth[top:bottom, left:right]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def backproject(u, v, depth, fx, fy, cx, cy):
    """Optical-frame (x, y, z) of a pixel at ``depth`` metres."""
    return ((u - cx) * depth / fx,
            (v - cy) * depth / fy,
            float(depth))


def pixel_ray(u, v, fx, fy, cx, cy):
    """Direction through a pixel in the optical frame."""
    return ((u - cx) / fx, (v - cy) / fy, 1.0)


def transform_point(transform, point):
    """Apply a TransformStamped to a point in its source frame."""
    rotation = transform.transform.rotation
    translation = transform.transform.translation
    rotated = rotate_vector(
        (rotation.x, rotation.y, rotation.z, rotation.w), point)
    return (rotated[0] + translation.x,
            rotated[1] + translation.y,
            rotated[2] + translation.z)


def quaternion_matrix(quaternion):
    """Return the 3x3 rotation matrix of an (x, y, z, w) quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


def transform_points(transform, points):
    """Apply a TransformStamped to an N x 3 array of points."""
    rotation = transform.transform.rotation
    translation = transform.transform.translation
    matrix = quaternion_matrix(
        (rotation.x, rotation.y, rotation.z, rotation.w))
    return np.asarray(points) @ matrix.T + np.array(
        [translation.x, translation.y, translation.z])


def masked_camera_points(mask, depth, fx, fy, cx, cy, stride=TOP_FACE_STRIDE,
                         min_points=TOP_FACE_MIN_POINTS):
    """
    Camera-frame points for the valid depth pixels inside ``mask``.

    A strided sample keeps the array small; the cube still contributes over a
    thousand points at the look pose. None means there was no usable depth.
    """
    height, width = depth.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST)
    rows = np.arange(0, height, stride)
    columns = np.arange(0, width, stride)
    sampled = mask[np.ix_(rows, columns)]
    v, u = np.nonzero(sampled)
    if u.size == 0:
        return None
    u = (u * stride).astype(np.int64)
    v = (v * stride).astype(np.int64)
    z = np.asarray(depth[v, u], dtype=np.float64)
    valid = np.isfinite(z) & (z > 0.0)
    if int(valid.sum()) < min_points:
        return None
    u, v, z = u[valid], v[valid], z[valid]
    return np.column_stack((
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
    ))


def highest_face_center(points, tolerance=TOP_FACE_HEIGHT_TOLERANCE,
                        min_points=TOP_FACE_MIN_POINTS, closing_axis=None,
                        size=None, viewer=None):
    """
    Centre of the highest horizontal face in an N x 3 arm-frame point set.

    Called with the back-projected, transformed points of the red blob. The
    top face is the highest part of the cube, and because it is horizontal
    every point on it has the same z; averaging those points is not biased by
    the visible side faces, unlike the centroid's depth sample.

    The mean is only correct when the whole face is visible. The finger pad
    between the camera and the cube hides part of the face at the look pose,
    so ``closing_axis`` (the finger closing direction, unit or not), ``size``
    (cube edge) and ``viewer`` (the camera origin, same frame) recentre a
    clipped face from its far edge: the visible strip reaches the cube edge
    away from the camera, and the centre is half an edge back towards the
    camera. A face whose span along the axis is at least
    ``FULL_FACE_FRACTION`` of ``size`` is treated as complete and keeps the
    mean.
    """
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < min_points \
            or points.shape[1] != 3:
        return None
    top = float(np.max(points[:, 2]))
    kept = points[points[:, 2] >= top - float(tolerance)]
    if kept.shape[0] < min_points:
        return None
    centre = kept.mean(axis=0)
    if closing_axis is not None and size is not None and viewer is not None:
        axis = np.asarray(closing_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        edge = float(size)
        if norm > 1e-9 and edge > 0.0:
            axis = axis / norm
            projection = kept @ axis
            low = float(projection.min())
            high = float(projection.max())
            if high - low < FULL_FACE_FRACTION * edge:
                viewer_edge = float(np.dot(
                    np.asarray(viewer, dtype=np.float64), axis))
                # The occluded end is the one nearest the camera, so the real
                # cube edge is the projection extreme farthest from it.
                if viewer_edge >= 0.5 * (low + high):
                    far_edge, towards_viewer = low, 1.0
                else:
                    far_edge, towards_viewer = high, -1.0
                wanted = far_edge + towards_viewer * edge / 2.0
                centre = centre + (wanted - float(centre @ axis)) * axis
    return (float(centre[0]), float(centre[1]), float(centre[2]))


def transform_direction(transform, direction):
    """Rotate a direction (no translation) into the target frame."""
    rotation = transform.transform.rotation
    return rotate_vector(
        (rotation.x, rotation.y, rotation.z, rotation.w), direction)


def point_on_plane(origin, direction, plane_z):
    """Intersect a ray with the horizontal plane z = ``plane_z``."""
    if abs(direction[2]) < 1e-9:
        return None
    distance = (plane_z - origin[2]) / direction[2]
    if distance <= 0.0:
        return None
    return (origin[0] + distance * direction[0],
            origin[1] + distance * direction[1],
            float(plane_z))


def hand_position_for_surface(surface, size, tcp_offset):
    """
    Hand origin for a top grasp of a cube whose top face is at ``surface``.

    The camera sees the top face, not the cube centre, so half the cube edge
    is subtracted first; the hand then sits ``tcp_offset`` above the centre,
    which is where the palm clears the cube.
    """
    return (surface[0], surface[1],
            surface[2] - float(size) / 2.0 + float(tcp_offset))


def lifted_position(grasp, height):
    """Grasp hand pose raised by ``height`` metres."""
    return (grasp[0], grasp[1], grasp[2] + float(height))


class VisualGraspNode(MobileManipulationNode):
    """Grasp the red cube at the pose measured by the wrist camera."""

    def __init__(self, node_name='visual_grasp'):
        super().__init__(node_name)
        self.image_topic = str(self.declare_parameter(
            'image_topic', IMAGE_TOPIC).value)
        self.camera_info_topic = str(self.declare_parameter(
            'camera_info_topic', CAMERA_INFO_TOPIC).value)
        self.depth_topic = str(self.declare_parameter(
            'depth_topic', DEPTH_TOPIC).value)
        self.min_blob_area = float(self.declare_parameter(
            'min_blob_area', float(MIN_BLOB_AREA)).value)
        self.detection_timeout = float(self.declare_parameter(
            'detection_timeout', DETECTION_TIMEOUT).value)
        self.look_position = (
            float(self.declare_parameter(
                'look_x', LOOK_POSITION[0]).value),
            float(self.declare_parameter(
                'look_y', LOOK_POSITION[1]).value),
            float(self.declare_parameter(
                'look_z', LOOK_POSITION[2]).value),
        )

        self.color_image = None
        self.image_frame = None
        self.camera_info = None
        self.depth = None
        # Cube top-face point in the world frame. The cube does not move
        # before the grasp, so this is the stable measurement; panda_link0
        # coordinates are recomputed from it whenever the base has crept.
        self.surface_world = None
        self.debug_pub = self.create_publisher(
            Image, DEBUG_IMAGE_TOPIC, 10)
        self.create_subscription(
            Image, self.image_topic, self._on_color_image,
            qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)

    def _on_color_image(self, message):
        try:
            self.color_image = bgr_image(message)
            self.image_frame = message.header.frame_id
        except ValueError as error:
            self.get_logger().warning('忽略图像：%s' % error)

    def _on_camera_info(self, message):
        self.camera_info = message

    def _on_depth(self, message):
        try:
            self.depth = depth_image(message)
        except ValueError as error:
            self.get_logger().warning('忽略深度图：%s' % error)

    def wait_for_camera_info(self, timeout=CAMERA_TIMEOUT):
        """Wait for the first CameraInfo message."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.camera_info is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.camera_info is not None

    def wait_for_color_image(self, timeout=CAMERA_TIMEOUT):
        """Wait for the first RGB image."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.color_image is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.color_image is not None

    def wait_for_depth(self, timeout=DEPTH_TIMEOUT):
        """Wait for the first depth image; False means fall back to a plane."""
        deadline = time.monotonic() + timeout
        while rclpy.ok() and self.depth is None \
                and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.depth is not None

    def detect_once(self):
        """Run the colour threshold on the latest image."""
        if self.color_image is None:
            return None
        return detect_red_blob(self.color_image, self.min_blob_area)

    def wait_for_detection(self, timeout=None):
        """Spin until a red blob is visible; None if the timeout expires."""
        if timeout is None:
            timeout = self.detection_timeout
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            detection = self.detect_once()
            if detection is not None:
                return detection
            rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def publish_debug_image(self, detection):
        """Publish the live image with the detected blob circled."""
        if self.color_image is None:
            return
        annotated = self.color_image.copy()
        if detection is not None:
            centre = (int(round(detection.u)), int(round(detection.v)))
            cv2.circle(annotated, centre, 10, (0, 255, 0), 2)
            cv2.drawMarker(annotated, centre, (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.image_frame or ''
        message.height, message.width = annotated.shape[:2]
        message.encoding = 'bgr8'
        message.is_bigendian = 0
        message.step = message.width * 3
        message.data = annotated.tobytes()
        self.debug_pub.publish(message)

    def lookup_camera_transform(self, timeout=TF_TIMEOUT):
        """Latest camera -> panda_link0 transform, or None."""
        deadline = time.monotonic() + timeout
        frame = self.image_frame or 'camera_optical_link'
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                return self.tf_buffer.lookup_transform(
                    ARM_BASE_FRAME, frame, Time())
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def target_plane_height(self):
        """Height of the cube's top face in panda_link0, for the fallback."""
        top = self.grasp['z'] + self.grasp['size'] / 2.0
        if self.ground_truth is not None:
            # The base only yaws, so x/y of the point do not change its z.
            return self.point_in_arm_frame((0.0, 0.0, top)).z

        transform = self.tf_buffer.lookup_transform(
            ARM_BASE_FRAME, ODOM_FRAME, Time())
        point = PointStamped()
        point.header.frame_id = ODOM_FRAME
        point.point.z = top
        return do_transform_point(point, transform).point.z

    def depth_at(self, u, v):
        """Depth in metres under an RGB pixel, or None if unavailable."""
        if self.depth is None or self.color_image is None:
            return None
        depth_u, depth_v = scale_pixel(
            u, v, self.color_image.shape, self.depth.shape)
        return depth_from_patch(self.depth, depth_u, depth_v)

    def closing_axis_in_arm_frame(self):
        """Finger closing direction (hand +y) in panda_link0."""
        return rotate_vector(
            self.orientation_in_arm_frame(GRIPPER_DOWN), (0.0, 1.0, 0.0))

    def target_point_in_arm_frame(self, detection, closing_axis=None):
        """
        Return ((x, y, z) in panda_link0, source).

        With a depth image the whole red blob is back-projected and the
        highest horizontal face is averaged; that is the cube's top face
        centre, without the side-face bias of a single centroid sample. If the
        depth is unusable, the pixel ray is intersected with the cube's known
        top-face height instead, so the grasp is still guided by the image.

        ``closing_axis`` is only passed by the grasp flow, where the gripper
        points down and its pads hide part of the top face: the measurement
        is then recentred from the visible far edge (see
        ``highest_face_center``). The search flow looks forward, where the
        pads are not over the cube, and keeps the plain mean.
        """
        if self.camera_info is None:
            raise RuntimeError('没有相机内参')
        fx, fy, cx, cy = camera_intrinsics(self.camera_info)
        transform = self.lookup_camera_transform()
        if transform is None:
            raise RuntimeError(
                '超时：没有 %s -> %s 的 TF'
                % (self.image_frame, ARM_BASE_FRAME))

        if self.depth is not None and self.color_image is not None:
            points = masked_camera_points(
                red_mask(self.color_image), self.depth, fx, fy, cx, cy)
            if points is not None:
                centre = highest_face_center(
                    transform_points(transform, points),
                    closing_axis=closing_axis,
                    size=self.grasp['size'],
                    viewer=transform_point(transform, (0.0, 0.0, 0.0)))
                if centre is not None:
                    return centre, 'top'

        depth = self.depth_at(detection.u, detection.v)
        if depth is not None:
            point = backproject(
                detection.u, detection.v, depth, fx, fy, cx, cy)
            return transform_point(transform, point), 'depth'

        origin = transform_point(transform, (0.0, 0.0, 0.0))
        direction = transform_direction(
            transform, pixel_ray(detection.u, detection.v, fx, fy, cx, cy))
        point = point_on_plane(origin, direction, self.target_plane_height())
        if point is None:
            raise RuntimeError('深度无效，且像素射线与目标高度平面没有交点')
        return point, 'plane'

    def look_at_pickup_area(self, orientation):
        """Move the wrist camera over the expected pickup spot."""
        self.get_logger().info(
            '把相机移到观察位姿 (%0.3f, %0.3f, %0.3f) [%s] ...'
            % (self.look_position[0], self.look_position[1],
               self.look_position[2], ARM_BASE_FRAME))
        return self.send_move_goal(
            make_reach_goal(self.look_position, orientation))

    def measure_target(self):
        """
        Detect the cube and return its top-face point in panda_link0.

        Raises RuntimeError when no red blob appears.
        """
        detection = self.wait_for_detection()
        if detection is None:
            raise RuntimeError(
                '%.0f s 内没有在 %s 中找到红色方块'
                % (self.detection_timeout, self.image_topic))
        self.publish_debug_image(detection)
        self.get_logger().info(
            '检测到红色方块：像素 (%0.1f, %0.1f)，面积 %0.0f px²'
            % (detection.u, detection.v, detection.area))

        point, source = self.target_point_in_arm_frame(
            detection, self.closing_axis_in_arm_frame())
        self.get_logger().info(
            '方块上表面在 %s 下为 (%0.4f, %0.4f, %0.4f)，来源：%s'
            % (ARM_BASE_FRAME, point[0], point[1], point[2],
               source_name(source)))
        self.remember_surface_world(point)
        return point

    def remember_surface_world(self, surface):
        """Store the measured top-face point in the world frame."""
        pose = self.base_pose()
        if pose is None:
            self.surface_world = None
            return None
        self.surface_world = point_in_world(surface, pose)
        self.get_logger().info(
            '方块上表面的世界坐标为 (%0.4f, %0.4f, %0.4f)；'
            '之后底盘再滑动也会用这个世界坐标重新计算抓取点'
            % self.surface_world)
        return self.surface_world

    def compensated_surface(self, fallback, stage):
        """Cube point in the current panda_link0 frame after base motion."""
        if self.surface_world is None or self.base_pose() is None:
            return fallback
        point = self.point_in_arm_frame(self.surface_world)
        current = (point.x, point.y, point.z)
        shift = math.hypot(
            current[0] - fallback[0], current[1] - fallback[1])
        if shift > 0.001:
            self.get_logger().info(
                '%s：底盘移动了 %.1f mm，按世界坐标把方块目标修正到 '
                '(%0.4f, %0.4f, %0.4f) [%s]'
                % (stage, shift * 1000.0, current[0], current[1],
                   current[2], ARM_BASE_FRAME))
        return current

    def refine_target(self, previous):
        """Try a closer detection once the hand is above the cube."""
        detection = self.wait_for_detection(REFINE_TIMEOUT)
        if detection is None:
            self.get_logger().warning(
                '近距离重新检测失败，沿用第一次测量的坐标')
            return self.compensated_surface(previous, '近距离检测失败')

        self.publish_debug_image(detection)
        point, source = self.target_point_in_arm_frame(
            detection, self.closing_axis_in_arm_frame())
        pose = self.base_pose()
        if self.surface_world is not None and pose is not None:
            candidate_world = point_in_world(point, pose)
            shift = max(
                abs(candidate_world[index] - self.surface_world[index])
                for index in range(3))
            if shift > MAX_REFINEMENT_SHIFT:
                self.get_logger().warning(
                    '近距离测量的世界坐标跳变 %.1f cm，可能检到了机械臂本身，'
                    '沿用第一次坐标' % (shift * 100.0))
                return self.compensated_surface(previous, '近距离测量跳变')
            self.surface_world = candidate_world
            self.get_logger().info(
                '近距离修正：(%0.4f, %0.4f, %0.4f)，来源：%s，'
                '世界坐标修正量 %.1f cm'
                % (point[0], point[1], point[2],
                   source_name(source), shift * 100.0))
            return point

        shift = max(abs(point[0] - previous[0]),
                    abs(point[1] - previous[1]))
        if shift > MAX_REFINEMENT_SHIFT:
            self.get_logger().warning(
                '近距离测量跳变 %.1f cm，可能检到了机械臂本身，沿用第一次坐标'
                % (shift * 100.0))
            return previous
        self.get_logger().info(
            '近距离修正：(%0.4f, %0.4f, %0.4f)，来源：%s，修正量 %.1f cm'
            % (point[0], point[1], point[2],
               source_name(source), shift * 100.0))
        return point

    def run(self):
        """Detect the red cube with the wrist camera and grasp it."""
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

        return self.grasp_after_arrival()

    def grasp_after_arrival(self):
        """
        Run steps 4-10: measure the cube with the wrist camera and grasp it.

        Split out of run() so an orchestrator that has already positioned the
        base (search_and_grasp drives here with Nav2) can reuse the whole
        grasp without the fixed 1 m forward drive.
        """
        if not self.wait_for_ground_truth():
            self.get_logger().warning(
                '没有收到地面真值里程计，退回轮式里程计 TF（抓取可能有偏差）')
            if not self.wait_for_base_tf():
                self.get_logger().error(
                    '超时：没有 %s -> %s 的 TF'
                    % (ODOM_FRAME, ARM_BASE_FRAME))
                return False

        # The arm reaches over a 0.6 m deep table. Put it in the planning
        # scene so MoveIt does not route the wrist or fingers through it;
        # without it Gazebo is the only thing that would notice a collision.
        self.publish_collision_objects([self.make_table_object()])

        self.get_logger().info('等待相机和相机内参 ...')
        if not self.wait_for_camera_info():
            self.get_logger().error(
                '超时：没有收到 %s' % self.camera_info_topic)
            return False
        if not self.wait_for_color_image():
            self.get_logger().error(
                '超时：没有收到 %s' % self.image_topic)
            return False
        if not self.wait_for_depth():
            self.get_logger().warning(
                '没有收到 %s，改用已知的方块上表面高度估算深度；'
                '抓取精度会下降，建议检查深度相机是否加载'
                % self.depth_topic)

        gripper_orientation = self.orientation_in_arm_frame(GRIPPER_DOWN)
        self.get_logger().info(
            '按底盘实际朝向补偿夹爪姿态：q=(%0.3f, %0.3f, %0.3f, %0.3f)'
            % gripper_orientation)

        self.get_logger().info('第四步：移动相机观察拾取区域 ...')
        if not self.look_at_pickup_area(gripper_orientation):
            self.get_logger().error('移动到观察位姿失败')
            return False

        self.get_logger().info('第五步：颜色分割并反投影 ...')
        try:
            surface = self.measure_target()
        except RuntimeError as error:
            self.get_logger().error('视觉定位失败：%s' % error)
            return False

        # The arm's reaction on the unbraked chassis yaws the base while it
        # moves to the look pose. Re-read the base orientation after every
        # large motion, otherwise the pads stay rotated relative to the cube
        # and hit its corners instead of its faces.
        self._spin_for(SETTLE_AFTER_MOVE)
        gripper_orientation = self.orientation_in_arm_frame(GRIPPER_DOWN)
        self.get_logger().info(
            '伸爪前重新读取底盘朝向：q=(%0.3f, %0.3f, %0.3f, %0.3f)'
            % gripper_orientation)
        surface = self.compensated_surface(surface, '伸爪前')

        size = self.grasp['size']
        tcp_offset = self.grasp['tcp_offset']
        grasp = hand_position_for_surface(surface, size, tcp_offset)
        pre_grasp = lifted_position(
            grasp, size / 2.0 + self.grasp['approach_clearance'])
        box_centre = (surface[0], surface[1], surface[2] - size / 2.0)
        box_object = make_box_object(
            box_centre, (size, size, size), object_id=BOX_OBJECT_ID)
        self.publish_collision_objects([box_object])

        self.get_logger().info(
            '第六步：伸到方块上方，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (pre_grasp[0], pre_grasp[1], pre_grasp[2], ARM_BASE_FRAME))
        if not self.send_move_goal(
                make_reach_goal(pre_grasp, gripper_orientation)):
            self.get_logger().error('伸向方块上方失败')
            return False

        surface = self.refine_target(surface)

        # The chassis is unbraked and can creep while the arm folds down to
        # the pre-grasp. Rebuild the grasp from the world-frame measurement
        # after the motion; otherwise a 1 cm base slide leaves the pads beside
        # the cube instead of around it.
        self._spin_for(SETTLE_AFTER_MOVE)
        surface = self.compensated_surface(surface, '下探前')
        grasp = hand_position_for_surface(surface, size, tcp_offset)
        corrected_pre_grasp = lifted_position(
            grasp, size / 2.0 + self.grasp['approach_clearance'])
        shift = max(abs(corrected_pre_grasp[0] - pre_grasp[0]),
                    abs(corrected_pre_grasp[1] - pre_grasp[1]))
        if shift > 0.002:
            self.get_logger().info(
                '方块相对夹爪移动了 %.1f cm，先重新移到方块正上方再下探 ...'
                % (shift * 100.0))
            gripper_orientation = self.orientation_in_arm_frame(GRIPPER_DOWN)
            if not self.send_move_goal(make_reach_goal(
                    corrected_pre_grasp, gripper_orientation)):
                self.get_logger().error('重新对准方块上方失败')
                return False
            self._spin_for(SETTLE_AFTER_MOVE)
            surface = self.compensated_surface(surface, '重新对准后')
            grasp = hand_position_for_surface(surface, size, tcp_offset)

        box_centre = (surface[0], surface[1], surface[2] - size / 2.0)
        box_object = make_box_object(
            box_centre, (size, size, size), object_id=BOX_OBJECT_ID)
        self.publish_collision_objects([box_object])
        gripper_orientation = self.orientation_in_arm_frame(GRIPPER_DOWN)
        self.get_logger().info(
            '下探前重新读取底盘朝向：q=(%0.3f, %0.3f, %0.3f, %0.3f)'
            % gripper_orientation)
        # The hand is about to pass the cube with the pads a few millimetres
        # away; one pad that drifted shut would catch the top edge. Verify the
        # open gap, and push the fingers against the stops if it fell short.
        if not self.ensure_gripper_open('下探前'):
            self.get_logger().error('下探前夹爪没有完全张开')
            return False

        self.get_logger().info(
            '第七步：垂直下探到方块中心，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (grasp[0], grasp[1], grasp[2], ARM_BASE_FRAME))
        if not self.send_cartesian_path(
                [make_pose(grasp, gripper_orientation)]):
            self.get_logger().error('下探到方块中心失败')
            return False

        # The pads descend fully open, so they nominally clear each face by
        # 15 mm. If the base slid further during the descend, closing would
        # hit the cube's top edge and push it sideways; stop while the cube is
        # still untouched.
        surface = self.compensated_surface(surface, '下探后')
        lateral = math.hypot(
            surface[0] - grasp[0], surface[1] - grasp[1])
        lateral_y = abs(surface[1] - grasp[1])
        self.get_logger().info(
            '下探后夹爪中心与方块中心的水平偏差为 %.1f mm（沿手指方向 '
            '%.1f mm，允许 %.1f mm）。'
            % (lateral * 1000.0, lateral_y * 1000.0,
               PRE_CLOSE_CLEARANCE * 1000.0))
        if lateral > PRE_CLOSE_CLEARANCE:
            self.get_logger().error(
                '方块偏离夹爪中心 %.1f mm，超过 %.1f mm 的预闭合间隙；'
                '此时闭爪会撞到方块边缘。已中止，避免把方块顶到一侧后'
                '捏在它的棱上。请重跑，让底盘停稳后再下探。'
                % (lateral * 1000.0, PRE_CLOSE_CLEARANCE * 1000.0))
            return False

        # Close the position gap down to a few millimetres on the near face
        # before the force stage. The press then only has to travel that
        # little, so it seats the pads instead of accelerating them into the
        # cube, which is what dragged the cube sideways in earlier runs.
        pre_close = pre_close_position_for(size, lateral_y)
        self.get_logger().info(
            '第八步：预闭合夹爪到 %.3f m，再用 %.1f N 的力夹紧 %.2f m 的方块 ...'
            % (pre_close, self.grasp['squeeze_effort'], size))
        self.publish_collision_objects([make_removed_object(BOX_OBJECT_ID)])
        if not self.pre_close_gripper(pre_close):
            self.get_logger().error('预闭合夹爪失败')
            return False
        if not self.squeeze_gripper():
            self.get_logger().error('夹紧方块失败')
            return False

        self.get_logger().info('第九步：把方块挂到夹爪上 ...')
        if not self.attach_box_in_gazebo():
            self.get_logger().error('抬起方块失败：方块没有固定到夹爪上')
            return False
        self.publish_attached_object(make_attached_object(box_object))
        # The weld carries the cube from here on; keep the 10 N press and the
        # pads chatter against the rigid cube (visible in the wrist camera).
        self.release_grip()

        lift = lifted_position(grasp, self.grasp['lift_height'])
        self.get_logger().info(
            '第十步：垂直向上抬起 %.0f cm，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (self.grasp['lift_height'] * 100.0,
               lift[0], lift[1], lift[2], ARM_BASE_FRAME))
        if not self.send_cartesian_path(
                [make_pose(lift, gripper_orientation)],
                velocity=self.grasp['lift_velocity']):
            self.get_logger().error('抬起方块失败')
            return False

        self.get_logger().info(
            '视觉抓取完成：方块已被夹爪提起 %.0f cm。'
            % (self.grasp['lift_height'] * 100.0))
        return True


def main(args=None):
    """Entry point."""
    rclpy.init(args=args)
    node = VisualGraspNode()
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
