import yaml

class TaskBase():
    def __init__(self, num_joints, robot_type, yaml_config) -> None:
        with open(yaml_config) as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc
        self.num_joints = num_joints
        self.robot_type = robot_type

    def compute_control(self, position, velocity, sensors):
        pass
