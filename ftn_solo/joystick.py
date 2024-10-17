#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

class JoyMessage(Node):
    def __init__(self):
        super().__init__("joy_message_node")
        self.subscriber = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.publisher = self.create_publisher(Float32MultiArray, 'acceleration_change', 10)

    def joy_callback(self, msg):
        selected_axes = [msg.axes[0], msg.axes[1], msg.axes[3], msg.axes[4]]
        selected_buttons = [float(msg.buttons[4]), float(msg.buttons[5])] 
        long_msg = Float32MultiArray()
        long_msg.data =  selected_axes + selected_buttons
        self.publisher.publish(long_msg)


def main(args=None):
    pass
    # rclpy.init()
    # node = JoyMessage()
    # rclpy.spin(node)
    # rclpy.shutdown()


if __name__ == '__main__':
    main()

# button order in Joy array
# 4 = L1
# 5 = R1
# 6 = L2 also axes 2 (unpressed position value = 1.0, pressed position value = -1.0)
# 7 = R2 also axes 5

# thumbstick order in Joy axis array (values from 1.0 ot -1.0)
# 0 = left-right (1 to -1) for left thumbstick
# 1 = up-down for left thumbstick
# 3 = left-right for right thumbstick
# 4 = up-down for right thumbstick