import numpy as np
from rclpy.node import Node


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class FeedbackLinearization():
    def __init__(self, robot, yaml_config) -> None:
        self.node = Node("FL")
        self.robot = robot
        num_joints = robot.nv - 6
        self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        self.B = float_or_list(yaml_config["B"], num_joints)
        self.Fv = float_or_list(yaml_config["Fv"], num_joints)
        self.friction_cutoff = float_or_list(
            yaml_config["friction_cutoff"], num_joints)
        self.max_control = float_or_list(
            yaml_config["max_control"], num_joints)

    def compute_control(self,  ref_position,  ref_velocity, ref_acceleration, position, velocity):
        friction_velocity = np.where(
            abs(velocity) > self.friction_cutoff, velocity, 0)
        qa = ref_acceleration + self.Kp * (ref_position - position) + self.Kd * (ref_velocity -
                                                                                 velocity)
        control = np.dot(self.robot.data.M[6:, 6:], qa) + self.robot.data.nle[6:] + \
            self.B * velocity + self.Fv * np.sign(friction_velocity)
        return np.clip(control, -self.max_control, self.max_control)
