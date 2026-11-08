"""Tests for the grasp cube configuration and its Gazebo SDF models."""

import os
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from my_robot_sim.grasp_box import (
    PALM_DEPTH,
    box_pose,
    box_sdf,
    load_config,
    pedestal_dimensions,
    pedestal_height,
    pedestal_pose,
    pedestal_sdf,
)

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG = os.path.join(PKG_ROOT, 'config', 'grasp_box.yaml')


def write_config(directory, **overrides):
    """Copy the real config, apply overrides and return the new path."""
    with open(CONFIG, encoding='utf-8') as handle:
        document = yaml.safe_load(handle)
    document['grasp_box'].update(overrides)
    path = os.path.join(str(directory), 'grasp_box.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(document, handle)
    return path


def test_config_describes_a_5cm_cube_beyond_the_chassis():
    """The cube must be reachable after the 1 m drive, not inside the robot."""
    config = load_config(CONFIG)
    assert config['size'] == pytest.approx(0.05)
    assert config['mass'] > 0.0
    assert config['x'] > 1.3
    assert config['z'] == pytest.approx(0.25)
    assert config['pedestal_size'] > config['size']
    assert box_pose(config) == (config['x'], config['y'], config['z'])


def test_tcp_offset_clears_the_palm():
    """The palm would press on the cube if the grasp point were too high."""
    config = load_config(CONFIG)
    assert config['tcp_offset'] > PALM_DEPTH + config['size'] / 2.0


def test_pedestal_top_meets_the_cube_bottom():
    """A free cube at z = 0.25 would fall, so a pedestal supports it."""
    config = load_config(CONFIG)
    bottom = config['z'] - config['size'] / 2.0
    assert pedestal_height(config) == pytest.approx(bottom)
    assert pedestal_pose(config) == pytest.approx(
        (config['x'], config['y'], bottom / 2.0))
    assert pedestal_dimensions(config) == pytest.approx(
        (config['pedestal_size'], config['pedestal_size'], bottom))


def test_box_sdf_is_a_dynamic_cube_with_grasp_friction():
    """The cube must be liftable and visibly orange in gzclient."""
    config = load_config(CONFIG)
    model = ET.fromstring(box_sdf(config)).find('model')
    assert model.get('name') == 'grasp_box'
    assert model.find('static') is None

    link = model.find('link')
    size = [float(value) for value in
            link.find('collision/geometry/box/size').text.split()]
    assert size == pytest.approx([config['size']] * 3)

    mass = float(link.find('inertial/mass').text)
    assert mass == pytest.approx(config['mass'])
    inertia = mass * config['size'] * config['size'] / 6.0
    for axis in ('ixx', 'iyy', 'izz'):
        assert float(link.find('inertial/inertia/%s' % axis).text) == \
            pytest.approx(inertia)

    assert float(link.find(
        'collision/surface/friction/ode/mu').text) == \
        pytest.approx(config['friction'])
    assert float(link.find(
        'collision/surface/friction/ode/mu2').text) == \
        pytest.approx(config['friction'])
    assert link.find('visual/material/script/name').text == 'Gazebo/Orange'


def test_box_sdf_names_match_the_attach_command():
    """The Gazebo plugin addresses the cube as grasp_box::link."""
    model = ET.fromstring(box_sdf(load_config(CONFIG))).find('model')
    assert model.get('name') == 'grasp_box'
    links = model.findall('link')
    assert len(links) == 1
    assert links[0].get('name') == 'link'


def test_box_is_never_auto_disabled():
    """A cube resting on the pedestal must stay simulated to be picked up."""
    config = load_config(CONFIG)
    model = ET.fromstring(box_sdf(config)).find('model')
    assert model.findtext('allow_auto_disable').strip() == 'false'


def test_pedestal_sdf_is_static_and_reaches_the_cube():
    """The pedestal top face must be exactly at the cube's bottom face."""
    config = load_config(CONFIG)
    model = ET.fromstring(pedestal_sdf(config)).find('model')
    assert model.get('name') == 'grasp_pedestal'
    assert model.findtext('static').strip() == 'true'
    size = [float(value) for value in
            model.find('link/collision/geometry/box/size').text.split()]
    assert size == pytest.approx(pedestal_dimensions(config))
    assert size[2] == pytest.approx(config['z'] - config['size'] / 2.0)


def test_sdf_models_are_accepted_by_gazebo(tmp_path):
    """Malformed SDF makes spawn_entity fail silently in the launch."""
    config = load_config(CONFIG)
    for name, builder in (('grasp_box', box_sdf),
                          ('grasp_pedestal', pedestal_sdf)):
        path = tmp_path / (name + '.sdf')
        path.write_text(builder(config), encoding='utf-8')
        result = subprocess.run(
            ['gz', 'sdf', '-p', str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_load_config_rejects_a_palm_collision(tmp_path):
    """tcp_offset has to be deeper than the palm plus half the cube."""
    path = write_config(tmp_path, tcp_offset=PALM_DEPTH + 0.05 / 2.0)
    with pytest.raises(ValueError, match='tcp_offset'):
        load_config(path)


def test_config_uses_a_bounded_squeeze_effort():
    """The constant grasp force must be positive and within the joint limit."""
    config = load_config(CONFIG)
    assert 0.0 < config['squeeze_effort'] <= 20.0


def test_load_config_rejects_a_squeeze_effort_above_the_joint_limit(tmp_path):
    path = write_config(tmp_path, squeeze_effort=25.0)
    with pytest.raises(ValueError, match='squeeze_effort'):
        load_config(path)


def test_config_uses_a_valid_lift_velocity():
    """The lift is a MoveIt scaling factor, so 0 < lift_velocity <= 1."""
    config = load_config(CONFIG)
    assert 0.0 < config['lift_velocity'] <= 1.0


def test_load_config_rejects_a_lift_velocity_above_one(tmp_path):
    path = write_config(tmp_path, lift_velocity=1.5)
    with pytest.raises(ValueError, match='lift_velocity'):
        load_config(path)


def test_load_config_rejects_missing_and_invalid_values(tmp_path):
    with open(CONFIG, encoding='utf-8') as handle:
        document = yaml.safe_load(handle)
    del document['grasp_box']['mass']
    path = os.path.join(str(tmp_path), 'missing.yaml')
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(ValueError, match='mass'):
        load_config(path)

    path = write_config(tmp_path, mass=-1.0)
    with pytest.raises(ValueError, match='mass'):
        load_config(path)

    path = write_config(tmp_path, size=0.0)
    with pytest.raises(ValueError, match='size'):
        load_config(path)
