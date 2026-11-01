import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SquareMover(Node):
    def __init__(self):
        super().__init__('square_mover')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

    def run_square(self):
        msg = Twist()
        # 让车先停一秒，确保发布器连上
        time.sleep(1.0) 

        for i in range(4):
            # 1. 直行 2 秒
            self.get_logger().info(f'第 {i+1} 边：直行')
            msg.linear.x = 0.3
            msg.angular.z = 0.0
            for _ in range(20):
                self.publisher_.publish(msg)
                time.sleep(0.1)

            # 2. 转弯 1 秒
            self.get_logger().info(f'第 {i+1} 次：转弯')
            msg.linear.x = 0.0
            msg.angular.z = 1.57 # 90度/秒
            for _ in range(10):
                self.publisher_.publish(msg)
                time.sleep(0.1)

        # 3. 停车
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info('正方形走完，停车！')

def main(args=None):
    rclpy.init(args=args)
    node = SquareMover()
    node.run_square()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()