import yaml

class TaskBase():
    def __init__(self, num_joints, robot_type, yaml_config) -> None:
        self.config = yaml_config
        self.num_joints = num_joints
        self.robot_type = robot_type

    def compute_control(self, position, velocity, sensors):
        pass
