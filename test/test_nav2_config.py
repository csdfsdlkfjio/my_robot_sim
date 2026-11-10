"""Nav2 参数文件的静态检查."""

import os

import pytest
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


def test_amcl_uses_the_beam_model_next_to_the_long_table():
    """
    A likelihood field treats points inside the 0.6 m thick table as perfect.

    With the base parked at the lateral standoff, scan endpoints that fall
    inside the table are at distance zero from an occupied cell, so AMCL
    happily slides 0.45 m east into the table (observed: vision placed the
    cube at 1.72 m while Gazebo had it at 1.45 m). The beam model raycasts
    each beam and penalises the ones that would have crossed the table.
    """
    amcl = load_params()['amcl']['ros__parameters']
    assert amcl['laser_model_type'] == 'beam'
    # The two boxes must stay visible, otherwise the 8 x 8 m room is
    # symmetric under 90 degree turns.
    assert amcl['max_beams'] == 360


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


def test_goal_checker_keeps_xy_tight_but_relaxes_yaw():
    """
    The stuck-at-goal logs showed 0.00 m remaining, so xy was never the issue.

    The wrist camera looks almost straight down and only sees a +/-8 cm patch
    at the cube top, so xy must stay tight; the base yaw is compensated by
    the arm, so the 0.10 rad yaw tolerance can be relaxed to stop the
    end-of-approach oscillation.
    """
    server = load_params()['controller_server']['ros__parameters']
    checker = server['general_goal_checker']
    assert 0.03 <= checker['xy_goal_tolerance'] <= 0.08
    assert checker['yaw_goal_tolerance'] >= 0.20
    # DWB's own goal test must not stop the base before the goal checker
    # is satisfied.
    assert server['FollowPath']['xy_goal_tolerance'] == pytest.approx(
        checker['xy_goal_tolerance'])


def test_launch_passes_params_file():
    """nav2.launch.py 必须把自定义参数文件传进去."""
    with open(LAUNCH, encoding='utf-8') as handle:
        source = handle.read()
    assert "'params_file'" in source
    assert 'config' in source and 'nav2_params.yaml' in source
