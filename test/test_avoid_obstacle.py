"""avoid_obstacle 中纯逻辑部分的单元测试."""

import math
from types import SimpleNamespace

from my_robot_sim.avoid_obstacle import (
    CLEAR_DISTANCE,
    OBSTACLE_DISTANCE,
    choose_command,
    front_distance,
    front_sector_ranges,
)

SAMPLES = 360
ANGLE_MIN = -math.pi
ANGLE_INCREMENT = 2.0 * math.pi / SAMPLES


def make_scan(ranges, angle_min=ANGLE_MIN, angle_increment=ANGLE_INCREMENT):
    """构造一个最小可用的 LaserScan 替身，避免依赖 ROS 运行时."""
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
    )


def test_front_sector_detects_angle_zero_not_index_zero():
    """正前方（索引 N/2）的障碍必须被检测到."""
    ranges = [10.0] * SAMPLES
    ranges[SAMPLES // 2] = 0.3
    scan = make_scan(ranges)
    assert min(front_sector_ranges(scan)) == 0.3


def test_front_sector_ignores_rear_obstacle():
    """正后方（索引 0 和 -1）的障碍不能影响前方判断."""
    ranges = [10.0] * SAMPLES
    ranges[0] = 0.2
    ranges[-1] = 0.2
    scan = make_scan(ranges)
    assert min(front_sector_ranges(scan)) == 10.0


def test_front_sector_filters_invalid_values():
    """inf、NaN 和 0 都应被过滤掉."""
    ranges = [float('inf')] * SAMPLES
    center = SAMPLES // 2
    ranges[center - 2:center + 3] = [0.0, float('nan'), 0.8, float('inf'), 5.0]
    scan = make_scan(ranges)
    assert front_sector_ranges(scan) == [0.8, 5.0]


def test_front_distance_ignores_single_outlier():
    """单个 range_min 坏点不能把前方判成危险."""
    ranges = [2.0] * SAMPLES
    ranges[SAMPLES // 2] = 0.1
    scan = make_scan(ranges)
    assert front_distance(scan) == 2.0


def test_front_distance_keeps_real_obstacle():
    """真障碍物占满扇区时，第 3 小的距离仍然很近."""
    ranges = [10.0] * SAMPLES
    ranges[SAMPLES // 2 - 3:SAMPLES // 2 + 4] = [0.5] * 7
    scan = make_scan(ranges)
    assert front_distance(scan) == 0.5


def test_front_distance_returns_inf_without_valid_data():
    """全部是 inf 时返回 inf（表示前方空旷）."""
    scan = make_scan([float('inf')] * SAMPLES)
    assert front_distance(scan) == float('inf')


def test_choose_command_starts_turning_when_obstacle_is_close():
    """障碍物进入阈值内时应停车并转向."""
    command, turning = choose_command(OBSTACLE_DISTANCE - 0.1, False)
    assert command.linear.x == 0.0
    assert command.angular.z > 0.0
    assert turning is True


def test_choose_command_keeps_turning_until_clear():
    """迟滞：距离略大于开始阈值但仍不够空旷时必须继续转向."""
    command, turning = choose_command(OBSTACLE_DISTANCE + 0.1, True)
    assert command.linear.x == 0.0
    assert command.angular.z > 0.0
    assert turning is True


def test_choose_command_stops_turning_when_clear():
    """前方超过 CLEAR_DISTANCE 后才恢复直行."""
    command, turning = choose_command(CLEAR_DISTANCE + 0.1, True)
    assert command.linear.x > 0.0
    assert command.angular.z == 0.0
    assert turning is False


def test_choose_command_moves_forward_when_clear():
    """前方安全时应直行."""
    command, turning = choose_command(CLEAR_DISTANCE + 1.0, False)
    assert command.linear.x > 0.0
    assert command.angular.z == 0.0
    assert turning is False
