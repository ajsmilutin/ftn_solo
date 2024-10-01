#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32

class JoyMessage(Node):
    def __init__(self):
        super().__init__("joy_message_node")
        self.subscriber = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.publisher = self.create_publisher(Float32, 'acceleration_change', 10)

    def joy_callback(self, msg):
        self.publisher.publish(Float32(data = msg.axes[0]))


def main(args=None):
    rclpy.init()
    node = JoyMessage()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()