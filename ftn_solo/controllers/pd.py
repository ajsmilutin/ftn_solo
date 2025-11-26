import numpy as np


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class PDWithFrictionCompensation():
    def __init__(self, robot, yaml_config) -> None:
        num_joints = robot.nv - 6
        self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        self.B = float_or_list(yaml_config["B"], num_joints)
        self.Fv = float_or_list(yaml_config["Fv"], num_joints)
        self.sigma = float_or_list(
            yaml_config["sigma"], num_joints)
        self.max_control = float_or_list(
            yaml_config["max_control"], num_joints)

    def clip(self, control):
        return np.clip(control, -self.max_control, self.max_control)

    def compute_pd(self, ref_position, ref_velocity, ref_acceleration, position, velocity):
        return self.Kp * (ref_position - position) + self.Kd * (ref_velocity - velocity)

    def compute_friction(self, ref_position, ref_velocity, ref_acceleration, position, velocity):
        return self.B * ref_velocity + self.Fv * np.arctan(ref_velocity / self.sigma) * 2/np.pi

    def compute_control(self, ref_position, ref_velocity, ref_acceleration, position, velocity):
        return self.clip(self.compute_pd(ref_position, ref_velocity, ref_acceleration, position, velocity) +
                         self.compute_friction(ref_position, ref_velocity, ref_acceleration, position, velocity))


class PDWithFrictionAndGravityCompensation(PDWithFrictionCompensation):
    def __init__(self, robot, yaml_config):
        super().__init__(robot, yaml_config)
        self.robot = robot

    def compute_control(self, ref_position, ref_velocity, ref_acceleration, position, velocity):
        return self.clip(self.compute_pd(ref_position, ref_velocity, ref_acceleration, position, velocity) +
                         self.compute_friction(
                             ref_position, ref_velocity, ref_acceleration, position, velocity)
                         + self.robot.data.nle[6:]
                         )
