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
    highest_face_center,
    lifted_position,
    masked_camera_points,
    pixel_ray,
    point_on_plane,
    quaternion_matrix,
    red_mask,
    scale_pixel,
    source_name,
    transform_direction,
    transform_point,
    transform_points,
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


def test_quaternion_matrix_rotates_about_z():
    """A 90 degree yaw maps the camera x axis to the arm y axis."""
    matrix = quaternion_matrix(quaternion_yaw(math.pi / 2.0))
    rotated = matrix @ np.array([1.0, 0.0, 0.0])
    assert rotated == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert quaternion_matrix((0.0, 0.0, 0.0, 0.0)) == pytest.approx(
        np.eye(3))


def test_transform_points_matches_the_single_point_transform():
    """The array version must apply the same rotation and translation."""
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 3.0
    (transform.transform.rotation.x,
     transform.transform.rotation.y,
     transform.transform.rotation.z,
     transform.transform.rotation.w) = quaternion_yaw(math.pi / 2.0)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    transformed = transform_points(transform, points)
    assert transformed[0] == pytest.approx(
        transform_point(transform, (1.0, 0.0, 0.0)), abs=1e-9)
    assert transformed[1] == pytest.approx(
        transform_point(transform, (0.0, 1.0, 0.0)), abs=1e-9)


def test_masked_camera_points_backprojects_the_masked_pixels():
    """Only masked pixels with a valid depth contribute camera points."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 3] = 255
    mask[2, 1] = 255
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[1, 3] = 0.5
    depth[2, 1] = 0.0  # invalid; must be dropped
    points = masked_camera_points(
        mask, depth, 100.0, 100.0, 2.0, 2.0, stride=1, min_points=1)
    assert points is not None
    assert points.shape == (1, 3)
    # u=3, v=1 -> x=(3-2)*0.5/100, y=(1-2)*0.5/100, z=0.5
    assert points[0] == pytest.approx((0.005, -0.005, 0.5), abs=1e-9)


def test_masked_camera_points_reject_an_empty_mask():
    """No masked depth means no measurement, not a made-up point."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    depth = np.full((4, 4), 0.5, dtype=np.float32)
    assert masked_camera_points(
        mask, depth, 100.0, 100.0, 2.0, 2.0, stride=1, min_points=1) is None


def test_highest_face_center_ignores_the_side_faces():
    """The top face is flat and highest; its centre is not pulled sideways."""
    xs = np.linspace(0.420, 0.470, 21)
    ys = np.linspace(-0.020, 0.030, 21)
    top = np.array([[x, y, 0.040] for x in xs for y in ys])
    # A thin sliver of the +y side face: its top rows reach the top height.
    side = np.array([[x, 0.030, z] for x in xs for z in (0.036, 0.039)])
    centre = highest_face_center(np.vstack([side, top]))
    assert centre is not None
    # The side face's top strip is inside the height band, so the mean z is
    # a fraction of a millimetre below the plane; that is far below the
    # 2.6 cm error of the old centroid-depth sample.
    assert centre[2] == pytest.approx(0.040, abs=0.001)
    assert centre[0] == pytest.approx(float(np.mean(xs)), abs=0.002)
    assert centre[1] == pytest.approx(float(np.mean(ys)), abs=0.004)


def test_highest_face_center_recentres_a_face_clipped_by_the_pad():
    """
    The pad hides the near half of the top face at the look pose.

    Only the far strip is visible. Its mean sits 15 mm towards the camera;
    anchoring on the far edge must recover the real square's centre.
    """
    xs = np.linspace(0.420, 0.470, 21)
    ys = np.linspace(-0.025, -0.005, 11)  # far half; the near half is hidden
    strip = np.array([[x, y, 0.275] for x in xs for y in ys])
    centre = highest_face_center(
        strip, closing_axis=(0.0, 1.0, 0.0), size=0.05,
        viewer=(0.45, 0.05, 0.45))
    assert centre is not None
    assert centre[1] == pytest.approx(0.0, abs=1e-9)
    assert centre[0] == pytest.approx(0.445, abs=1e-9)
    assert centre[2] == pytest.approx(0.275, abs=1e-9)
    # Without the corrective arguments the old, biased mean is still what
    # the search flow gets.
    plain = highest_face_center(strip)
    assert plain[1] == pytest.approx(-0.015, abs=1e-9)


def test_highest_face_center_recentres_from_either_viewer_side():
    """The far edge is chosen by the viewer's side, not hard-coded."""
    xs = np.linspace(0.420, 0.470, 21)
    ys = np.linspace(0.005, 0.025, 11)  # far half for a viewer at -y
    strip = np.array([[x, y, 0.275] for x in xs for y in ys])
    centre = highest_face_center(
        strip, closing_axis=(0.0, 1.0, 0.0), size=0.05,
        viewer=(0.45, -0.05, 0.45))
    assert centre[1] == pytest.approx(0.0, abs=1e-9)


def test_highest_face_center_keeps_the_mean_for_a_full_face():
    """A complete (or yawed) face keeps the plain mean, uneven density and all."""
    xs = np.linspace(0.420, 0.470, 21)
    ys = np.linspace(-0.025, 0.025, 41)
    face = np.array([[x, y, 0.275] for x in xs for y in ys])
    # Overweight one edge; the mean must be the plain mean, not the anchor.
    dense = np.vstack([face, np.tile([0.445, 0.025, 0.275], (30, 1))])
    plain = highest_face_center(dense)
    corrected = highest_face_center(
        dense, closing_axis=(0.0, 1.0, 0.0), size=0.05,
        viewer=(0.45, 0.05, 0.45))
    assert corrected == pytest.approx(plain)
    assert corrected[1] > 0.0  # the overweighted edge still pulls the mean


def test_highest_face_center_needs_enough_points():
    """A handful of stray pixels must not become a grasp target."""
    points = np.array([[0.0, 0.0, 0.04]] * 3)
    assert highest_face_center(points, min_points=10) is None
    assert highest_face_center(None) is None


def test_source_name_labels_the_top_face_measurement():
    """The logs must say which measurement produced the grasp point."""
    assert source_name('top') == '顶面点云'
    assert source_name('depth') == '深度图'
    assert source_name('plane') == '已知高度平面'


def test_hand_position_moves_from_the_visible_face_to_the_cube_centre():
    """The camera sees the top face; the hand target is in the cube centre."""
    hand = hand_position_for_surface((0.5, 0.1, 0.275), 0.05, 0.10)
    assert hand == pytest.approx((0.5, 0.1, 0.35))


def test_lifted_position_raises_only_z():
    """A lift must not move the grasp sideways."""
    lifted = lifted_position((0.5, 0.1, 0.35), 0.10)
    assert lifted == pytest.approx((0.5, 0.1, 0.45))
