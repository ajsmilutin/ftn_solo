from transitions import Machine
import yaml
import numpy as np
from scipy.interpolate import CubicSpline


class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray((len(self.durations), num_joints),dtype=np.float64)        
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i,:] = poses[pose_name]

class ControllerJointSpline():
    states = ["start", "follow_spline"]

    def __init__(self, num_of_joints,  config_yaml) -> None:
        with open(config_yaml) as stream:
            try:
                data = yaml.safe_load(stream)
            except Exception as exc:
                raise exc
        controller = data["controller"]
        self.Kp = float(controller["Kp"])
        self.Kd = float(controller["Kd"])
        self.B = np.array(controller["B"], dtype=np.float64)
        self.Fv = np.array(controller["Fv"], dtype=np.float64)
        self.friction_cutoff = float(controller["friction_cutoff"])
        self.max_control = float(controller["max_control"])
        self.parse_poses(data["poses"])
        self.on_start = SplineData(
            data["on_start"], num_of_joints, self.poses)
        self.loop = []
        for point in data["loop"]:
            self.loop.append(SplineData(point, num_of_joints, self.poses))
        self.loop_phase = 0
        self.machine = Machine(
            model=self, states=ControllerJointSpline.states, initial="start")
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
        friction_velocity = np.where(
            abs(self.ref_velocity) > self.friction_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(friction_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def compute_control(self, t, q, qv, sensors):
        self.tick(t, q, qv)
        return self.control
