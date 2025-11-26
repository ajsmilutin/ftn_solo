import yaml
from ftn_solo_control import SplineTrajectory
from ftn_solo.utils.trajectories import SplineData
from ftn_solo.controllers import PDWithFrictionCompensation
import numpy as np
from robot_properties_solo import Solo12Robot, UnitreeGo2Robot, AnymalRobot
from ftn_solo_control import FixedRobotEstimator


class TaskBase:
    def __init__(self, num_joints, robot_type, yaml_config) -> None:
        self.config = yaml_config
        self.num_joints = num_joints
        self.robot_type = robot_type

    def compute_control(self, position, velocity, sensors):
        pass


class TaskWithInitPose(TaskBase):
    def __init__(self, num_joints, robot_type, yaml_config) -> None:
        super().__init__(num_joints, robot_type, yaml_config)
        if robot_type == "solo12":
            self.robot = Solo12Robot()
        elif robot_type == "unitree_go2":
            self.robot = UnitreeGo2Robot()
        elif robot_type == "anymal":
            self.robot = AnymalRobot()
        else:
            raise ("Only solo12, unitree_go2 and anymal supported")
        self.parse_poses(self.config["poses"])
        self.on_start = SplineData(
            self.config["on_start"], self.num_joints, self.poses)
        self.step = 0
        self.joint_controller = PDWithFrictionCompensation(
            self.robot.pin_robot, self.config["joint_controller"]
        )
        self.estimator = FixedRobotEstimator(
            0.001, self.robot.pin_robot.model, self.robot.pin_robot.data, True, np.array([0, 0, 0.4]), np.eye(3))

    def parse_poses(self, poses):
        self.poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(
                poses[pose_name], dtype=np.float64)

    def init_pose(self, q, qv):
        self.compute_trajectory(0, q, self.on_start)

    def compute_trajectory(self, tstart, q, sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = SplineTrajectory(False)
        self.trajectory.add(q, 0)
        for t, pose in zip(sequence.durations, sequence.poses):
            self.trajectory.add(pose, t)
        self.trajectory.set_start(tstart)
        self.last_pose = sequence.poses[-1, :]

    def following_spline(self, t, q, qv):
        self.ref_position, self.ref_velocity, self.ref_acceleration = (
            self.trajectory.get(t)
        )
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, self.ref_acceleration, q, qv
        )
        return t >= self.transition_end
