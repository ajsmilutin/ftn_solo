from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import PDWithFrictionCompensation

class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray((len(self.durations), num_joints),dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i,:] = poses[pose_name]

class TaskJointSpline(TaskBase):
    states = ["start", "follow_spline"]

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        super().__init__(num_joints, robot_type, config_yaml)
        self.joint_controller = PDWithFrictionCompensation(self.num_joints, self.config["joint_controller"])
        self.parse_poses(self.config["poses"])
        self.on_start = SplineData(
            self.config["on_start"], self.num_joints, self.poses)
        self.loop = []
        for point in self.config["loop"]:
            self.loop.append(SplineData(point, self.num_joints, self.poses))
        self.loop_phase = 0
        self.machine = Machine(
            model=self, states=TaskJointSpline.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "follow_spline", conditions="following_spline")
        self.machine.add_transition(
            "tick", "follow_spline", "follow_spline", conditions="following_spline")

        self.machine.on_enter_follow_spline(self.compute_spline)

    def parse_poses(self, poses):
        self.poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(poses[pose_name], dtype=np.float64)

    def init_pose(self, q, qv):
        self.compute_trajectory(0, q, self.on_start)

    def compute_trajectory(self, tstart, q,  sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = CubicSpline(np.hstack(
            ([tstart], tstart + sequence.durations)), np.vstack((q, sequence.poses)), bc_type="clamped")
        self.last_pose = sequence.poses[-1, :]

    def compute_spline(self, t, q, qv):
        self.compute_trajectory(t, self.last_pose, self.loop[self.loop_phase])
        self.loop_phase = (self.loop_phase+1) % len(self.loop)

    def following_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.control = self.joint_controller.compute_control(self.ref_position, self.ref_velocity, q, qv)
        return t >= self.transition_end

    def compute_control(self, t, q, qv, sensors):
        self.tick(t, q, qv)
        return self.control
