"""Nav2 参数文件的静态检查."""

import os

import yaml

PKG_ROOT = os.path.join(os.path.dirname(__file__), '..')
PARAMS = os.path.join(PKG_ROOT, 'config', 'nav2_params.yaml')
LAUNCH = os.path.join(PKG_ROOT, 'launch', 'nav2.launch.py')


def load_params():
    """读取 Nav2 参数文件."""
    with open(PARAMS, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_amcl_uses_base_link():
    """AMCL 必须用 base_link：本车 TF 树里没有 base_footprint."""
    amcl = load_params()['amcl']['ros__parameters']
    assert amcl['base_frame_id'] == 'base_link'
    assert amcl['use_sim_time'] is True


def test_costmaps_use_base_link():
    """代价地图与机器人的实际 base frame 保持一致."""
    params = load_params()
    for node in ('global_costmap', 'local_costmap'):
        section = params[node][node]['ros__parameters']
        assert section['robot_base_frame'] == 'base_link'


def iter_values(node):
    """递归取出 YAML 里所有的叶子值."""
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_values(value)
    else:
        yield node


def test_no_base_footprint_value():
    """参数值里不应再出现 base_footprint（注释不算）."""
    assert 'base_footprint' not in list(iter_values(load_params()))


def test_launch_passes_params_file():
    """nav2.launch.py 必须把自定义参数文件传进去."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    assert "'params_file'" in source
    assert 'config' in source and 'nav2_params.yaml' in source
