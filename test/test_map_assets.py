"""地图资源与演示世界文件的静态检查."""

import os
import xml.etree.ElementTree as ET

import yaml

PKG_ROOT = os.path.join(os.path.dirname(__file__), '..')
MAP_YAML = os.path.join(PKG_ROOT, 'maps', 'my_map.yaml')
DEMO_WORLD = os.path.join(PKG_ROOT, 'worlds', 'demo.world')


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
