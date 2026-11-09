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
    TF_TIMEOUT,
    MobileManipulationNode,
    make_attached_object,
    make_box_object,
    make_gripper_goal,
    make_pose,
    make_reach_goal,
    make_removed_object,
    pre_close_position,
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

# A closer re-detection is only trusted if it does not jump further than
# this; a wildly different blob is more likely an arm part than the cube.
MAX_REFINEMENT_SHIFT = 0.05  # m

Detection = namedtuple('Detection', ['u', 'v', 'area'])


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

    def __init__(self):
        super().__init__('visual_grasp')
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

    def target_point_in_arm_frame(self, detection):
        """
        Return ((x, y, z) in panda_link0, 'depth' or 'plane').

        With a depth image the pixel is back-projected directly. Without one,
        the ray is intersected with the cube's known top-face height; the
        pixel still decides where in the plane the cube is, so the grasp is
        still guided by the image.
        """
        if self.camera_info is None:
            raise RuntimeError('没有相机内参')
        fx, fy, cx, cy = camera_intrinsics(self.camera_info)
        transform = self.lookup_camera_transform()
        if transform is None:
            raise RuntimeError(
                '超时：没有 %s -> %s 的 TF'
                % (self.image_frame, ARM_BASE_FRAME))

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

        point, source = self.target_point_in_arm_frame(detection)
        self.get_logger().info(
            '方块上表面在 %s 下为 (%0.4f, %0.4f, %0.4f)，来源：%s'
            % (ARM_BASE_FRAME, point[0], point[1], point[2],
               '深度图' if source == 'depth' else '已知高度平面'))
        return point

    def refine_target(self, previous):
        """Try a closer detection once the hand is above the cube."""
        detection = self.wait_for_detection(REFINE_TIMEOUT)
        if detection is None:
            self.get_logger().warning(
                '近距离重新检测失败，沿用第一次测量的坐标')
            return previous
        self.publish_debug_image(detection)
        point, source = self.target_point_in_arm_frame(detection)
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
               '深度图' if source == 'depth' else '已知高度平面',
               shift * 100.0))
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

        self.get_logger().info('第三步：向前行驶 ...')
        if not self.drive_forward():
            self.stop_and_settle()
            return False
        self.stop_and_settle()

        if not self.wait_for_ground_truth():
            self.get_logger().warning(
                '没有收到地面真值里程计，退回轮式里程计 TF（抓取可能有偏差）')
            if not self.wait_for_base_tf():
                self.get_logger().error(
                    '超时：没有 %s -> %s 的 TF'
                    % (ODOM_FRAME, ARM_BASE_FRAME))
                return False

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

        refined = self.refine_target(surface)
        if refined != surface:
            surface = refined
            grasp = hand_position_for_surface(surface, size, tcp_offset)
            box_centre = (surface[0], surface[1], surface[2] - size / 2.0)
            box_object = make_box_object(
                box_centre, (size, size, size), object_id=BOX_OBJECT_ID)
            self.publish_collision_objects([box_object])

        self._spin_for(SETTLE_AFTER_MOVE)
        gripper_orientation = self.orientation_in_arm_frame(GRIPPER_DOWN)
        self.get_logger().info(
            '下探前重新读取底盘朝向：q=(%0.3f, %0.3f, %0.3f, %0.3f)'
            % gripper_orientation)

        self.get_logger().info(
            '第七步：垂直下探到方块中心，手的目标为 '
            '(%0.3f, %0.3f, %0.3f) [%s] ...'
            % (grasp[0], grasp[1], grasp[2], ARM_BASE_FRAME))
        if not self.send_cartesian_path(
                [make_pose(grasp, gripper_orientation)]):
            self.get_logger().error('下探到方块中心失败')
            return False

        self.get_logger().info(
            '第八步：预闭合夹爪到 %.3f m，再用 %.1f N 的力夹紧 %.2f m 的方块 ...'
            % (pre_close_position(size), self.grasp['squeeze_effort'], size))
        self.publish_collision_objects([make_removed_object(BOX_OBJECT_ID)])
        if not self.send_move_goal(
                make_gripper_goal(pre_close_position(size))):
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
