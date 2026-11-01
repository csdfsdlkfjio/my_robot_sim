"""RViz 配置的静态检查，确保显示项指向正确的话题."""

import os

import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'rviz', 'slam.rviz')


def load_manager():
    """读取 rviz 配置中的 Visualization Manager 段."""
    with open(CONFIG_PATH, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    return config['Visualization Manager']


def displays_by_class(suffix):
    """按插件类名后缀筛选显示项."""
    displays = load_manager()['Displays']
    return [display for display in displays
            if display['Class'].endswith(suffix)]


def test_fixed_frame_is_map():
    """SLAM 场景下固定坐标系必须是 map."""
    assert load_manager()['Global Options']['Fixed Frame'] == 'map'


def test_laserscan_subscribes_to_scan():
    """激光雷达（LaserScan）显示必须订阅 /scan."""
    scans = displays_by_class('/LaserScan')
    assert scans, '缺少 LaserScan 显示'
    assert any(scan['Topic']['Value'] == '/scan' for scan in scans)


def test_map_subscribes_to_map():
    """Map 显示必须订阅 /map."""
    maps = displays_by_class('/Map')
    assert maps, '缺少 Map 显示'
    assert any(item['Topic']['Value'] == '/map' for item in maps)
