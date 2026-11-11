"""地图资源与演示世界文件的静态检查."""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from my_robot_sim.grasp_box import (
    load_config,
    table_dimensions,
    table_pose,
)
from my_robot_sim.search_and_grasp import (
    NAV_GOAL_CLEARANCE,
    NAV_GOAL_RETREAT_STEPS,
    STANDOFF_DISTANCE,
    clearance_violation,
    standoff_candidates,
    standoff_goal,
)

PKG_ROOT = os.path.join(os.path.dirname(__file__), '..')
MAP_YAML = os.path.join(PKG_ROOT, 'maps', 'my_map.yaml')
DEMO_WORLD = os.path.join(PKG_ROOT, 'worlds', 'demo.world')
GRASP_CONFIG = os.path.join(PKG_ROOT, 'config', 'grasp_box.yaml')


def load_map_info():
    """读取地图 yaml."""
    with open(MAP_YAML, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def read_pgm_header(path):
    """解析二进制 PGM（P5）的头，返回 (宽, 高)."""
    with open(path, 'rb') as handle:
        magic = handle.readline().strip()
        size = handle.readline().split()
    assert magic == b'P5'
    return int(size[0]), int(size[1])


def test_map_yaml_has_required_fields():
    """map_server 需要的字段必须齐全，且单位合理."""
    info = load_map_info()
    for key in ('image', 'resolution', 'origin', 'mode'):
        assert key in info, f'缺少字段 {key}'
    assert info['mode'] == 'trinary'
    assert info['resolution'] > 0.0
    assert len(info['origin']) == 3
    assert info['occupied_thresh'] > info['free_thresh']


def test_map_image_exists_and_is_valid():
    """地图 yaml 引用的图片必须存在，且是合法的 P5 PGM."""
    info = load_map_info()
    image_path = os.path.join(os.path.dirname(MAP_YAML), info['image'])
    assert os.path.isfile(image_path), f'找不到地图图片 {image_path}'
    width, height = read_pgm_header(image_path)
    assert width > 0 and height > 0
    assert os.path.getsize(image_path) > width * height


def read_world_box(name):
    """Return (pose, size) of a world model's box collision."""
    root = ET.parse(DEMO_WORLD).getroot()
    model = root.find('world/model[@name="%s"]' % name)
    assert model is not None, 'demo.world has no %s' % name
    pose = [float(value) for value in model.find('pose').text.split()]
    size = [float(value) for value in
            model.find('link/collision/geometry/box/size').text.split()]
    return pose, size


def test_world_table_matches_the_grasp_config():
    """
    The static world table and the MoveIt planning-scene table must agree.

    The table lives in demo.world so SLAM and Nav2 see it from the start;
    grasp_box.yaml repeats its numbers for the planning scene. If the two
    drift apart MoveIt would avoid a table that is not there and miss the
    one that is.
    """
    config = load_config(GRASP_CONFIG)
    pose, size = read_world_box('grasp_table')
    assert pose[:3] == pytest.approx(table_pose(config))
    assert pose[3:] == pytest.approx([0.0, 0.0, 0.0])
    assert size == pytest.approx(table_dimensions(config))
    # The cube must rest on the world table top, not float or sink.
    assert size[2] == pytest.approx(
        config['z'] - config['size'] / 2.0)


def test_standoff_clears_the_long_table():
    """
    The Nav2 observation pose must leave room for the 0.22 m footprint.

    The robot approaches the table along +x. The standoff goal is 0.6 m from
    the cube centre; the table's near face is 0.30 m from it, so the base
    centre is 0.30 m from the table and the footprint clears it.
    """
    config = load_config(GRASP_CONFIG)
    pose, size = read_world_box('grasp_table')
    near_face_x = pose[0] - size[0] / 2.0
    standoff_x = config['x'] - STANDOFF_DISTANCE
    assert standoff_x < near_face_x, 'the observation pose is inside the table'
    assert near_face_x - standoff_x >= 0.27, (
        'the 0.22 m footprint would touch the table')
    # The path from the robot start (0, 0) to the observation pose is along
    # -x of the table and must not cross it.
    assert 0.0 < standoff_x < near_face_x


def test_world_obstacle_does_not_overlap_the_table():
    """Moving box_1 must not leave it intersecting the new table."""
    table_pose_values, table_size = read_world_box('grasp_table')
    box_pose_values, box_size = read_world_box('box_1')

    def separated(axis):
        table_min = table_pose_values[axis] - table_size[axis] / 2.0
        table_max = table_pose_values[axis] + table_size[axis] / 2.0
        box_min = box_pose_values[axis] - box_size[axis] / 2.0
        box_max = box_pose_values[axis] + box_size[axis] / 2.0
        return box_max <= table_min or box_min >= table_max

    # Two AABBs are disjoint when they are separated on at least one axis.
    assert separated(0) or separated(1), 'box_1 overlaps the table'


def test_map_contains_the_table_and_boxes():
    """
    The new table must be occupied in the map Nav2 and AMCL load.

    Without it the planner would happily route the base through the table,
    and AMCL would score every lidar return from the table as a mismatch.
    The cube is spawned at runtime, so it is intentionally absent.
    """
    info = load_map_info()
    image = read_pgm(os.path.join(os.path.dirname(MAP_YAML), info['image']))
    for y in np.linspace(-0.9, 0.9, 7):
        for x in (1.20, 1.45, 1.70):
            assert map_pixel(image, info, x, y) == 0, (
                'table footprint at (%.2f, %.2f) is not occupied' % (x, y))
        assert map_pixel(image, info, 1.10, y) == 254, (
            'space west of the table is not free')
    for name in ('box_1', 'box_2'):
        pose, _ = read_world_box(name)
        assert map_pixel(image, info, pose[0], pose[1]) == 0, (
            '%s is not occupied in the map' % name)


def test_lateral_observation_goals_are_free_in_the_map():
    """
    Left / centre / right cube positions must all leave the goal free.

    ``standoff_goal`` puts the goal on the line from the cube to the robot,
    so a cube at y = +/-0.6 m gives a diagonal approach that runs close to
    the table's front face. Run the same check search_and_grasp runs, with
    the real map and the real clearance, instead of approximating it with
    separately rounded cell offsets.
    """
    config = load_config(GRASP_CONFIG)
    info = load_map_info()
    image = read_pgm(os.path.join(os.path.dirname(MAP_YAML), info['image']))
    grid = occupancy_grid_from_pgm(image, info)
    for cube_y in (-0.6, 0.0, 0.6):
        goal_x, goal_y, _ = standoff_goal(
            (config['x'], cube_y), (0.0, 0.0), STANDOFF_DISTANCE)
        reason = clearance_violation(
            grid, image.shape[1], image.shape[0],
            info['origin'][0], info['origin'][1], info['resolution'],
            goal_x, goal_y, NAV_GOAL_CLEARANCE)
        assert reason is None, (
            'goal at (%.2f, %.2f) is rejected for cube y=%.1f: %s'
            % (goal_x, goal_y, cube_y, reason))


def test_goal_retreat_recovers_from_a_biased_cube_estimate():
    """
    A cube at y=-0.6 m seen obliquely was estimated at (1.539, -0.642).

    That 9 cm east bias put the nominal 0.6 m standoff at x=0.99, only
    0.16 m from the table, so the goal check refused it. The candidate
    generator must find a goal the check accepts while keeping the cube
    within the arm's reach, otherwise search_and_grasp would give up on a
    perfectly graspable left/right cube.
    """
    info = load_map_info()
    image = read_pgm(os.path.join(os.path.dirname(MAP_YAML), info['image']))
    grid = occupancy_grid_from_pgm(image, info)
    cube_xy, robot_xy = (1.5387, -0.6418), (0.09, -0.03)
    nominal = standoff_goal(cube_xy, robot_xy, STANDOFF_DISTANCE)
    assert clearance_violation(
        grid, image.shape[1], image.shape[0],
        info['origin'][0], info['origin'][1], info['resolution'],
        nominal[0], nominal[1], NAV_GOAL_CLEARANCE) is not None
    chosen = None
    for offset, extra, x, y, yaw in standoff_candidates(
            cube_xy, robot_xy, STANDOFF_DISTANCE):
        reason = clearance_violation(
            grid, image.shape[1], image.shape[0],
            info['origin'][0], info['origin'][1], info['resolution'],
            x, y, NAV_GOAL_CLEARANCE)
        if reason is None:
            chosen = (offset, extra, x, y)
            break
    assert chosen is not None, (
        'no bearing/retreat candidate cleared the table')
    # Reach-preserving: panda_link0 measures ~4 cm more than base_link.
    assert chosen[1] <= max(NAV_GOAL_RETREAT_STEPS)
    assert math.hypot(chosen[2] - cube_xy[0], chosen[3] - cube_xy[1]) <= (
        STANDOFF_DISTANCE + max(NAV_GOAL_RETREAT_STEPS) + 1e-9)


def test_lateral_goal_beyond_the_table_reach_is_rejected():
    """
    A cube near the table's end must not put the goal inside the table.

    The current 0.63 m standoff clears the table even from the origin, so
    the guard is checked with a short 0.4 m standoff: its goal falls inside
    the table's clearance band and must be refused.
    """
    config = load_config(GRASP_CONFIG)
    info = load_map_info()
    image = read_pgm(os.path.join(os.path.dirname(MAP_YAML), info['image']))
    grid = occupancy_grid_from_pgm(image, info)
    goal_x, goal_y, _ = standoff_goal(
        (config['x'], 0.75), (0.0, 0.0), 0.4)
    reason = clearance_violation(
        grid, image.shape[1], image.shape[0],
        info['origin'][0], info['origin'][1], info['resolution'],
        goal_x, goal_y, NAV_GOAL_CLEARANCE)
    assert reason == '附近有墙壁或障碍物'
    # And the standoff actually used must keep the same cube clear.
    goal_x, goal_y, _ = standoff_goal(
        (config['x'], 0.75), (0.0, 0.0), STANDOFF_DISTANCE)
    assert clearance_violation(
        grid, image.shape[1], image.shape[0],
        info['origin'][0], info['origin'][1], info['resolution'],
        goal_x, goal_y, NAV_GOAL_CLEARANCE) is None


def test_demo_world_is_valid_sdf():
    """演示世界必须是合法 SDF，且包含墙和障碍物."""
    root = ET.parse(DEMO_WORLD).getroot()
    assert root.tag == 'sdf'
    world = root.find('world')
    assert world is not None
    models = world.findall('model')
    assert len(models) >= 5, '演示世界至少要有四面墙和两个障碍物'
    for model in models:
        assert model.find('link') is not None


def occupancy_grid_from_pgm(image, info):
    """把 PGM 转成 map_server 按本 yaml 阈值发布的 OccupancyGrid.data."""
    occupancy = (255.0 - image.astype(np.float64)) / 255.0
    data = np.where(
        occupancy > info['occupied_thresh'], 100,
        np.where(occupancy < info['free_thresh'], 0, -1))
    # OccupancyGrid 的行从 origin 角落向上增长，PGM 第 0 行在地图上方。
    return data[::-1].reshape(-1)


def read_pgm(path):
    """读取二进制 PGM（P5），返回高度 x 宽度的灰度数组（第 0 行在顶部）."""
    with open(path, 'rb') as handle:
        assert handle.readline().strip() == b'P5'
        line = handle.readline()
        while line.startswith(b'#'):
            line = handle.readline()
        width, height = (int(value) for value in line.split())
        assert int(handle.readline()) == 255
        data = np.frombuffer(handle.read(width * height), dtype=np.uint8)
    assert data.size == width * height
    return data.reshape(height, width)


def read_walls():
    """返回 [(名字, x, y, x 尺寸, y 尺寸)]，来自 demo.world 的静态墙体."""
    root = ET.parse(DEMO_WORLD).getroot()
    walls = []
    for model in root.find('world').findall('model'):
        name = model.get('name', '')
        if not name.startswith('wall_'):
            continue
        x, y = (float(value) for value in
                model.find('pose').text.split()[:2])
        size = model.find('.//collision/geometry/box/size').text.split()
        walls.append((name, x, y, float(size[0]), float(size[1])))
    return walls


def map_pixel(image, info, x, y):
    """世界坐标 (x, y) 处的 PGM 灰度值."""
    resolution = info['resolution']
    col = int(math.floor((x - info['origin'][0]) / resolution + 1e-9))
    row = int(math.floor((y - info['origin'][1]) / resolution + 1e-9))
    height, width = image.shape
    assert 0 <= col < width and 0 <= row < height
    # PGM 第 0 行在地图上方；map_server 的行号从 origin 角落向上增长。
    return int(image[height - 1 - row, col])


def test_demo_world_walls_form_a_closed_ring():
    """墙体必须在四角互相搭接，底盘才没有能开出去的缝."""
    walls = read_walls()
    assert {wall[0] for wall in walls} == {
        'wall_north', 'wall_south', 'wall_east', 'wall_west'}
    by_name = {wall[0]: wall for wall in walls}
    # 南北墙沿 x 方向长 8.2 m，东西墙沿 y 方向长 8.2 m：两端都越过
    # +/-4 m 的墙体中心线，四角因此重叠，没有缺口。
    for name in ('wall_north', 'wall_south'):
        assert by_name[name][3] / 2.0 > 4.0
    for name in ('wall_east', 'wall_west'):
        assert by_name[name][4] / 2.0 > 4.0


def test_map_bounds_contain_the_walls_with_unknown_outside():
    """地图范围必须盖住墙体外沿，墙外留一圈 unknown."""
    info = load_map_info()
    width, height = read_pgm_header(
        os.path.join(os.path.dirname(MAP_YAML), info['image']))
    resolution = info['resolution']
    min_x, min_y = info['origin'][0], info['origin'][1]
    assert min_x <= -4.05 and min_y <= -4.05
    assert min_x + width * resolution >= 4.05
    assert min_y + height * resolution >= 4.05


def test_map_pixels_align_with_the_demo_world_walls():
    """
    墙体在栅格里的位置必须和 demo.world 完全一致.

    墙厚 0.1 m、中心在 +/-4 m：内表面 +/-3.95、外表面 +/-4.05。地图里
    每条墙的中心线必须是 occupied，墙内 5 cm 是自由空间，墙外是 unknown。
    """
    info = load_map_info()
    image = read_pgm(os.path.join(os.path.dirname(MAP_YAML), info['image']))
    for x in np.linspace(-3.0, 3.0, 13):
        assert map_pixel(image, info, x, 4.0) == 0, '北墙没有占到栅格'
        assert map_pixel(image, info, x, -4.0) == 0, '南墙没有占到栅格'
        assert map_pixel(image, info, x, 3.90) == 254, '北墙内侧不是自由空间'
        assert map_pixel(image, info, x, -3.90) == 254, '南墙内侧不是自由空间'
        assert map_pixel(image, info, x, 4.10) == 205, '北墙外侧不是 unknown'
        assert map_pixel(image, info, x, -4.10) == 205, '南墙外侧不是 unknown'
    for y in np.linspace(-3.0, 3.0, 13):
        assert map_pixel(image, info, 4.0, y) == 0, '东墙没有占到栅格'
        assert map_pixel(image, info, -4.0, y) == 0, '西墙没有占到栅格'
        assert map_pixel(image, info, 3.90, y) == 254, '东墙内侧不是自由空间'
        assert map_pixel(image, info, -3.90, y) == 254, '西墙内侧不是自由空间'
        assert map_pixel(image, info, 4.10, y) == 205, '东墙外侧不是 unknown'
        assert map_pixel(image, info, -4.10, y) == 205, '西墙外侧不是 unknown'
