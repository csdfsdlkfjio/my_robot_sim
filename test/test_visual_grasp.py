"""Unit tests for the visual-grasp detection and geometry helpers."""

import math

import cv2
import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image

from my_robot_sim.visual_grasp import (
    Detection,
    backproject,
    bgr_image,
    camera_intrinsics,
    depth_from_patch,
    depth_image,
    detect_red_blob,
    hand_position_for_surface,
    lifted_position,
    pixel_ray,
    point_on_plane,
    red_mask,
    scale_pixel,
    transform_direction,
    transform_point,
)


def red_scene(width=320, height=240, centre=(160, 120), size=40):
    """Dark grey scene with one red square centred on ``centre``."""
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    half = size // 2
    left = max(0, centre[0] - half)
    right = min(width, centre[0] + half)
    top = max(0, centre[1] - half)
    bottom = min(height, centre[1] + half)
    image[top:bottom, left:right] = (0, 0, 255)
    return image


def image_message(array, encoding):
    """Wrap a uint8 array in a sensor_msgs/Image."""
    message = Image()
    message.height, message.width = array.shape[:2]
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = message.width * array.shape[2]
    message.data = array.tobytes()
    return message


def test_red_mask_finds_only_red_pixels():
    """The mask must select the red square and nothing else."""
    image = red_scene()
    mask = red_mask(image)
    assert mask.shape == image.shape[:2]
    assert mask[120, 160] == 255
    assert mask[10, 10] == 0


def test_detect_red_blob_returns_the_centroid():
    """The centroid must be the centre of the visible red area."""
    detection = detect_red_blob(red_scene(centre=(160, 120), size=40))
    assert isinstance(detection, Detection)
    assert detection.u == pytest.approx(160.0, abs=1.0)
    assert detection.v == pytest.approx(120.0, abs=1.0)
    assert detection.area == pytest.approx(40 * 40, rel=0.2)


def test_detect_red_blob_ignores_the_background():
    """A scene without red must not report a target."""
    image = np.full((240, 320, 3), 40, dtype=np.uint8)
    assert detect_red_blob(image) is None


def test_detect_red_blob_keeps_the_largest_blob():
    """Arm parts and reflections are small; the cube is the largest blob."""
    image = red_scene(centre=(80, 60), size=14)
    image[100:140, 200:240] = (0, 0, 255)
    detection = detect_red_blob(image)
    assert detection.u == pytest.approx(220.0, abs=1.0)
    assert detection.v == pytest.approx(120.0, abs=1.0)


def test_detect_red_blob_respects_the_minimum_area():
    """A few stray pixels must not count as the cube."""
    image = red_scene(centre=(160, 120), size=8)
    assert detect_red_blob(image, min_area=80.0) is None


def test_detect_red_blob_accepts_both_ends_of_the_hue_range():
    """Red wraps around the hue circle, so both ranges must be searched."""
    for hue in (2, 178):
        hsv = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv[20:60, 20:60] = (hue, 255, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        detection = detect_red_blob(image)
        assert detection is not None, 'hue %d was not detected' % hue
        assert detection.u == pytest.approx(40.0, abs=1.0)


def test_bgr_image_converts_rgb_encoding():
    """Gazebo publishes rgb8, OpenCV wants BGR."""
    message = image_message(
        np.array([[[10, 20, 30]]], dtype=np.uint8), 'rgb8')
    image = bgr_image(message)
    assert image[0, 0].tolist() == [30, 20, 10]


def test_bgr_image_keeps_bgr_encoding():
    """A bgr8 frame must not be swapped twice."""
    message = image_message(
        np.array([[[10, 20, 30]]], dtype=np.uint8), 'bgr8')
    assert bgr_image(message)[0, 0].tolist() == [10, 20, 30]


def test_bgr_image_rejects_unknown_encodings():
    """Mono images cannot be colour-thresholded."""
    message = image_message(
        np.array([[[10]]], dtype=np.uint8), 'mono8')
    with pytest.raises(ValueError):
        bgr_image(message)


def test_depth_image_reads_float_metres():
    """The Gazebo depth plugin publishes 32FC1 metres."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    message = Image()
    message.height, message.width = 2, 2
    message.encoding = '32FC1'
    message.is_bigendian = 0
    message.step = 8
    message.data = values.tobytes()
    depth = depth_image(message)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, values)


def test_depth_image_converts_millimetres():
    """A 16UC1 depth frame is millimetres and must become metres."""
    values = np.array([[1000, 2000]], dtype=np.uint16)
    message = Image()
    message.height, message.width = 1, 2
    message.encoding = '16UC1'
    message.is_bigendian = 0
    message.step = 4
    message.data = values.tobytes()
    depth = depth_image(message)
    np.testing.assert_allclose(depth, [[1.0, 2.0]])


def test_depth_from_patch_ignores_invalid_measurements():
    """Zeros and NaNs mean "no measurement"; the median must skip them."""
    depth = np.full((12, 12), np.nan, dtype=np.float32)
    depth[4:9, 4:9] = 0.5
    depth[6, 6] = 0.0
    depth[5, 5] = 2.0
    assert depth_from_patch(depth, 6.0, 6.0, radius=2) == pytest.approx(0.5)


def test_depth_from_patch_returns_none_when_everything_is_invalid():
    """No valid depth means the caller has to use the height fallback."""
    depth = np.zeros((10, 10), dtype=np.float32)
    assert depth_from_patch(depth, 5.0, 5.0) is None


def test_scale_pixel_maps_rgb_to_depth_resolution():
    """RGB and depth images need not have the same size."""
    assert scale_pixel(100.0, 50.0, (480, 640), (240, 320)) == \
        pytest.approx((50.0, 25.0))


def test_camera_intrinsics_reads_the_k_matrix():
    """fx, fy, cx, cy live in the K matrix of CameraInfo."""
    info = CameraInfo()
    info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    assert camera_intrinsics(info) == pytest.approx(
        (500.0, 500.0, 320.0, 240.0))


def test_backproject_returns_the_optical_frame_point():
    """Pixel + depth + intrinsics must give the expected camera ray point."""
    point = backproject(340.0, 260.0, 2.0, 500.0, 500.0, 320.0, 240.0)
    assert point == pytest.approx((0.08, 0.08, 2.0))


def test_pixel_ray_points_forward():
    """The optical frame looks along +z, so the ray's z is 1."""
    ray = pixel_ray(340.0, 260.0, 500.0, 500.0, 320.0, 240.0)
    assert ray == pytest.approx((0.04, 0.04, 1.0))


def test_point_on_plane_intersects_the_known_height():
    """Without depth, the pixel ray meets the cube's top-face plane."""
    point = point_on_plane((0.0, 0.0, 1.0), (0.5, 0.0, -1.0), 0.0)
    assert point == pytest.approx((0.5, 0.0, 0.0))


def test_point_on_plane_rejects_rays_that_miss():
    """A ray parallel to the plane, or pointing away, has no intersection."""
    assert point_on_plane((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.0) is None
    assert point_on_plane((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0) is None


def quaternion_yaw(angle):
    """Quaternion rotating about z by ``angle``."""
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


def test_transform_point_applies_rotation_then_translation():
    """TF transforms map camera points into the arm base frame."""
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 3.0
    (transform.transform.rotation.x,
     transform.transform.rotation.y,
     transform.transform.rotation.z,
     transform.transform.rotation.w) = quaternion_yaw(math.pi / 2.0)
    point = transform_point(transform, (1.0, 0.0, 0.0))
    assert point == pytest.approx((1.0, 3.0, 3.0), abs=1e-9)


def test_transform_direction_does_not_translate():
    """A ray direction only rotates when it enters the arm frame."""
    transform = TransformStamped()
    transform.transform.translation.x = 100.0
    (transform.transform.rotation.x,
     transform.transform.rotation.y,
     transform.transform.rotation.z,
     transform.transform.rotation.w) = quaternion_yaw(math.pi / 2.0)
    direction = transform_direction(transform, (1.0, 0.0, 0.0))
    assert direction == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_hand_position_moves_from_the_visible_face_to_the_cube_centre():
    """The camera sees the top face; the hand target is in the cube centre."""
    hand = hand_position_for_surface((0.5, 0.1, 0.275), 0.05, 0.10)
    assert hand == pytest.approx((0.5, 0.1, 0.35))


def test_lifted_position_raises_only_z():
    """A lift must not move the grasp sideways."""
    lifted = lifted_position((0.5, 0.1, 0.35), 0.10)
    assert lifted == pytest.approx((0.5, 0.1, 0.45))
