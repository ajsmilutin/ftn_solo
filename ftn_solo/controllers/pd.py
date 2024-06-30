import numpy as np


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class PDWithFrictionCompensation():
    def __init__(self, num_joints, yaml_config) -> None:
        self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        self.B = float_or_list(yaml_config["B"], num_joints)
        self.Fv = float_or_list(yaml_config["Fv"], num_joints)
        self.friction_cutoff = float_or_list(
            yaml_config["friction_cutoff"], num_joints)
        self.max_control = float_or_list(
            yaml_config["max_control"], num_joints)

    def compute_control(self, ref_position,  ref_velocity, position, velocity):
        friction_velocity = np.where(
            abs(ref_velocity) > self.friction_cutoff, ref_velocity, 0)
        control = self.Kp * (ref_position - position) + self.Kd * (ref_velocity -
                                                                   velocity) + self.B * ref_velocity + self.Fv * np.sign(friction_velocity)
        return np.clip(control, -self.max_control, self.max_control)
