"""激光雷达避障节点：正前方有障碍物时原地左转，直到前方足够空旷才继续直行."""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

FRONT_HALF_ANGLE = math.radians(30.0)  # 正前方 ±30° 视为安全走廊
OBSTACLE_DISTANCE = 0.6                # 小于该距离开始转向
CLEAR_DISTANCE = 1.0                   # 大于该距离才恢复直行（迟滞）
OUTLIER_RANK = 2                       # 取第 3 小的距离，过滤单点坏数据
FORWARD_SPEED = 0.3                    # 前进速度 m/s
TURN_SPEED = 1.0                       # 转向角速度 rad/s
SCAN_TIMEOUT = 1.0                     # 超过该时间没收到 /scan 就停车


def normalize_angle(angle):
    """把角度归一化到 [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def front_sector_ranges(msg, half_angle=FRONT_HALF_ANGLE):
    """
    返回正前方 ±half_angle 范围内所有有效距离.

    注意：不能假设 ranges[0] 是正前方。标准 LaserScan 的
    angle_min = -pi，索引 0 是正后方，正前方在索引 len/2 附近。
    这里按每个点的实际角度筛选，跟雷达的 angle_min 配置无关。
    """
    if not msg.ranges or msg.angle_increment <= 0.0:
        return []
    valid = []
    for index, distance in enumerate(msg.ranges):
        angle = msg.angle_min + index * msg.angle_increment
        if abs(normalize_angle(angle)) > half_angle:
            continue
        if math.isfinite(distance) and distance > 0.0:
            valid.append(distance)
    return valid


def front_distance(msg, half_angle=FRONT_HALF_ANGLE, rank=OUTLIER_RANK):
    """
    返回正前方第 rank+1 小的有效距离，没有有效数据时返回 inf.

    不直接取最小值：Gazebo 的射线雷达偶尔会给出一个 range_min 级别的
    单点坏值（日志里表现为 0.59 -> 0.10 -> 0.53），取第 3 小的距离可以
    把它过滤掉。真正的障碍物会占满整个扇区，不会被误滤。
    """
    valid = sorted(front_sector_ranges(msg, half_angle))
    if not valid:
        return float('inf')
    return valid[min(rank, len(valid) - 1)]


def choose_command(min_distance, turning):
    """
    带迟滞的避障决策，返回 (速度指令, 新的转向状态).

    进入转向后，必须等前方距离超过 CLEAR_DISTANCE 才恢复直行。
    否则车头稍微偏一点，障碍物就滑出扇区、读数变大，节点会误判为
    安全继续前进，结果一路蹭着障碍物走，最后顶到墙上。
    """
    if turning:
        if min_distance >= CLEAR_DISTANCE:
            turning = False
    elif min_distance < OBSTACLE_DISTANCE:
        turning = True

    command = Twist()
    if turning:
        command.angular.z = TURN_SPEED
    else:
        command.linear.x = FORWARD_SPEED
    return command, turning


class ObstacleAvoider(Node):
    """订阅 /scan，向 /cmd_vel 发布避障速度指令."""

    def __init__(self):
        super().__init__('obstacle_avoider')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.last_scan_time = None
        self.timeout_warned = False
        self.turning = False
        self.create_timer(0.5, self.watchdog)
        self.get_logger().info('避障节点已启动，等待雷达数据...')

    def laser_callback(self, msg):
        """根据正前方最近的障碍物决定前进还是转向."""
        self.last_scan_time = time.monotonic()
        if self.timeout_warned:
            self.get_logger().info('雷达数据已恢复')
            self.timeout_warned = False

        distance = front_distance(msg)
        command, self.turning = choose_command(distance, self.turning)
        if self.turning:
            self.get_logger().info(
                f'危险！前方障碍物 {distance:.2f} m，转向中')
        else:
            self.get_logger().info(
                f'前方安全，最近距离 {distance:.2f} m，继续前进')
        self.publisher_.publish(command)

    def watchdog(self):
        """超过 SCAN_TIMEOUT 没有雷达数据时停车，避免失控."""
        if (self.last_scan_time is not None
                and time.monotonic() - self.last_scan_time <= SCAN_TIMEOUT):
            return
        self.publisher_.publish(Twist())
        if not self.timeout_warned:
            self.get_logger().warn(
                f'超过 {SCAN_TIMEOUT:.0f} 秒没有收到 /scan，已停车')
            self.timeout_warned = True


def main(args=None):
    """启动节点并进入事件循环."""
    rclpy.init(args=args)
    node = ObstacleAvoider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
