import csv
from unittest.mock import NonCallableMagicMock

import numpy as np
from pyparsing import ParseException
from scipy.interpolate import CubicSpline
from scipy.signal import chirp
from transitions import Machine

from ftn_solo.controllers import PDWithFrictionCompensation

from .task_base import TaskBase


class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray(
            (len(self.durations), num_joints), dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i, :] = poses[pose_name]


class ChirpData:
    def __init__(self, yaml_config, chirp) -> None:
        self.duration = yaml_config["duration"]
        self.chirp = chirp[yaml_config["chirp"]]


class TaskJointSpline(TaskBase):
    states = ["start", "follow_spline", "idle"]

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        super().__init__(num_joints, robot_type, config_yaml)
        self.joint_controller = PDWithFrictionCompensation(
            self.num_joints, self.config["joint_controller"])
        self.parse_poses(self.config["poses"])

        if "chirp_gains" in self.config.keys():
            self.parse_chirp(self.config["chirp_gains"])
        if self.config["on_start"]["type"] == "spline":
            self.current_phase = SplineData(
                self.config["on_start"], self.num_joints, self.poses)
        elif self.config["on_start"]["type"] == "chirp":
            self.current_phase = ChirpData(
                self.config["on_start"], self.chirp_gains)
        if "chirp_torque" in self.config.keys():
            self.chirp_F0 = self.config["chirp_torque"]["F0"]
            self.chirp_F1 = self.config["chirp_torque"]["F1"]
        else:
            self.chirp_F0 = 0
            self.chirp_F1 = 0
        self.sequence = []
        self.loop = []
        if "loop" in self.config.keys():
            for point in self.config["loop"]:
                if point["type"] == "spline":
                    self.loop.append(SplineData(
                        point, self.num_joints, self.poses))
                elif point["type"] == "chirp":
                    self.loop.append(ChirpData(point, self.chirp_gains))
            self.loop_phase = 0
        elif "sequence" in self.config.keys():
            for point in self.config["sequence"]:
                if point["type"] == "spline":
                    self.sequence.append(SplineData(
                        point, self.num_joints, self.poses))
                elif point["type"] == "chirp":
                    self.sequence.append(ChirpData(point, self.chirp_gains))
            self.sequence_phase = 0
        else:
            raise (ParseException("No loop, or sequence defined."))
        self.machine = Machine(
            model=self, states=TaskJointSpline.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "follow_spline", conditions="following_spline")
        self.machine.add_transition(
            "tick", "follow_spline", "follow_spline", conditions="following_spline")
        self.machine.add_transition(
            "tick", "follow_spline", "idle", conditions="go_to_idle")
        self.machine.add_transition(
            'tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_follow_spline(self.compute_spline)
        self.machine.on_enter_idle(self.save_log)
        if "log_file" in self.config.keys():
            self.log_file = self.config["log_file"]
        else:
            self.log_file = None
        if (self.log_file is not None) and self.sequence:
            self.log_rows = []
        else:
            self.log_rows = None

    def parse_poses(self, poses):
        self.poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(
                poses[pose_name], dtype=np.float64)

    def parse_chirp(self, chirp_gains):
        self.chirp_gains = {}
        for chirp_name in chirp_gains:
            self.chirp_gains[chirp_name] = np.array(
                chirp_gains[chirp_name], dtype=np.float64)

    def init_pose(self, q, qv):
        if isinstance(self.current_phase, SplineData):
            self.compute_trajectory(0, q, self.current_phase)
        elif isinstance(self.current_phase, ChirpData):
            self.compute_chirp(0, self.current_phase)

    def compute_trajectory(self, tstart, q,  sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = CubicSpline(np.hstack(
            ([tstart], tstart + sequence.durations)), np.vstack((q, sequence.poses)), bc_type="clamped")
        self.last_pose = sequence.poses[-1, :]

    def compute_chirp(self, tstart, chirp_data):
        chirp_duration = chirp_data.duration
        self.transition_end = tstart + chirp_duration
        self.chirp_t = np.arange(
            0, chirp_duration + self.dt, self.dt / 2.0, dtype=np.float64)
        phase = 270
        self.chirp = chirp(self.chirp_t, self.chirp_F0,
                           chirp_duration, self.chirp_F1, phi=phase)

    def compute_spline(self, t, q, qv):
        if isinstance(self.current_phase, ChirpData):
            self.last_pose = q
        if self.loop:
            self.current_phase = self.loop[self.loop_phase]
            self.loop_phase = (self.loop_phase+1) % len(self.loop)
        elif self.sequence:
            self.current_phase = self.sequence[self.sequence_phase]
            self.sequence_phase = self.sequence_phase + 1
        if isinstance(self.current_phase, SplineData):
            self.compute_trajectory(
                t, self.last_pose, self.current_phase)
        elif isinstance(self.current_phase, ChirpData):
            self.compute_chirp(t, self.current_phase)

    def chirp_torque(self, t, q, qv):
        transition_start = self.transition_end - \
            self.current_phase.duration
        torques = self.current_phase.chirp
        try:
            chirp_index = np.where(self.chirp_t >= t - transition_start)[0][0]
            control = self.chirp[chirp_index] * torques
        except Exception as exc:
            raise exc
        self.ref_velocity = self.last_pose * 0.0
        self.ref_position = self.last_pose
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, q, qv)
        self.control = np.where(torques != 0.0, control, self.control)
        self.log_data(t, q, qv)

    def follow_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, q, qv)

    def following_spline(self, t, q, qv):
        if isinstance(self.current_phase, SplineData):
            self.follow_spline(t, q, qv)
        elif isinstance(self.current_phase, ChirpData):
            self.chirp_torque(t, q, qv)
        return (t >= self.transition_end) and not (self.go_to_idle(t, q, qv))

    def go_to_idle(self, t, q, qv):
        return (self.sequence) and (self.sequence_phase >= len(self.sequence)) and (t >= self.transition_end)

    def do_nothing(self, t, q, qv):
        return False

    def compute_control(self, t, q, qv, sensors):
        self.tick(t, q, qv)
        return self.control

    def log_data(self, t, q, qv):
        if self.log_rows is not None:
            row = [0.0] * (2 + 3 * self.num_joints)
            row[0] = self.sequence_phase
            row[1] = t
            start_index = 2
            end_index = self.num_joints + start_index
            row[start_index:end_index] = self.control.tolist()
            start_index = end_index
            end_index += self.num_joints
            row[start_index:end_index] = q.tolist()
            start_index = end_index
            end_index += self.num_joints
            row[start_index:end_index] = qv.tolist()
            self.log_rows.append(row)

    def save_log(self, t, q, qv):
        with open(self.log_file, "w") as log_file:
            log_csv = csv.writer(log_file)
            log_csv.writerows(self.log_rows)
            self.log_rows.clear()
