from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import chirp
from .task_base import TaskBase
from ftn_solo.controllers import *
import csv

class LoopData:
    def __init__(self, yaml_config, num_joints, poses, chirp_poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.chirp_durations = np.array(yaml_config["chirp_durations"]) 
        self.poses = np.ndarray((len(self.durations), num_joints),dtype=np.float64)
        self.chirp_poses = np.ndarray((len(self.durations), num_joints),dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i,:] = poses[pose_name]
            self.chirp_poses[i,:] = chirp_poses[pose_name]
            
            
            
class TaskFrictionIdent(TaskBase):
    states = ["start", "follow_spline", "friction_ident", "idle"]
    
    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        super().__init__(num_joints, robot_type, config_yaml)
        self.joint_controller = PDBase(self.num_joints, self.config["joint_controller"])
        self.parse_poses(self.config["poses"])
        self.on_start = LoopData(
            self.config["on_start"], self.num_joints, self.poses, self.chirp_poses)
        self.loop = []
        for point in self.config["loop"]:
            self.loop.append(LoopData(point, self.num_joints, self.poses, self.chirp_poses))
        self.loop_phase = -1
        self.dt = 0.001
        self.chirp_F0 = self.config["chirp_torque"]["F0"]
        self.chirp_F1 = self.config["chirp_torque"]["F1"]
        self.machine = Machine(
            model=self, states=TaskFrictionIdent.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "follow_spline", conditions="following_spline")
        self.machine.add_transition(
            "tick", "follow_spline", "friction_ident", conditions="following_spline")
        self.machine.add_transition(
            "tick", "friction_ident", "follow_spline", conditions="identification")        
        self.machine.add_transition(
            "tick", "follow_spline", "idle", conditions="go_to_idle")
        self.machine.add_transition('tick', 'idle', 'idle', conditions='do_nothing')

        self.machine.on_enter_follow_spline(self.compute_spline)
        self.machine.on_enter_friction_ident(self.compute_chirp)
        self.machine.on_enter_idle(self.save_log)
        self.log_rows = []
        self.log_file = self.config["log_file"]

    def parse_poses(self, poses):
        self.poses = {}
        self.chirp_poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(poses[pose_name]["q"], dtype=np.float64)
            self.chirp_poses[pose_name] = np.array(poses[pose_name]["chirp"], dtype=np.float64)

    def init_pose(self, q, qv):
        self.compute_trajectory(0, q, self.on_start)

    def compute_trajectory(self, tstart, q,  sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = CubicSpline(np.hstack(
            ([tstart], tstart + sequence.durations)), np.vstack((q, sequence.poses)), bc_type="clamped")
        self.last_pose = sequence.poses[-1, :]

    def compute_spline(self, t, q, qv):
        self.loop_phase = (self.loop_phase+1)
        self.compute_trajectory(t, q, self.loop[self.loop_phase])
        

    def following_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.control = self.joint_controller.compute_control(self.ref_position, self.ref_velocity, q, qv)
        return (self.loop_phase < len(self.loop) - 1) and (t >= self.transition_end)
    
    def compute_chirp(self, t, q ,qv):
        chirp_duration = self.loop[self.loop_phase].chirp_durations[-1]
        self.transition_end = t + chirp_duration
        self.chirp_t = np.arange(0, chirp_duration + self.dt, self.dt / 2.0, dtype = np.float64)
        phase = 270
        self.chirp = chirp(self.chirp_t, self.chirp_F0, chirp_duration, self.chirp_F1, phi = phase)
        
        
    def identification(self, t, q, qv):
        transition_start  = self.transition_end - self.loop[self.loop_phase].chirp_durations[-1]
        torques = self.loop[self.loop_phase].chirp_poses[-1, :]
        try:
            chirp_index = np.where(self.chirp_t >= t - transition_start)[0][0]
            control = self.chirp[chirp_index] * torques
        except Exception as exc:
            raise exc
        self.ref_velocity = self.loop[self.loop_phase].poses * 0.0
        self.ref_position = self.loop[self.loop_phase].poses
        self.control = self.joint_controller.compute_control(self.ref_position, self.ref_velocity, q, qv)
        self.control = np.where(torques != 0.0, control, self.control)
        self.log_data(t, q, qv)
        return (t >= self.transition_end)

    def go_to_idle(self, t, q, qv):
        return (self.loop_phase >= len(self.loop) - 1) and (t >= self.transition_end)

    def do_nothing(self, t, q, qv):
        self.control = self.on_start.poses
        return False

    def compute_control(self, t, q, qv, sensors):
        self.tick(t, q, qv)
        return self.control
    
    def log_data(self, t, q, qv):
        row = [0.0] * (2 + 3 * self.num_joints)
        row[0] = self.loop_phase
        row[1] = t
        start_index = 2
        end_index = self.num_joints + start_index
        row[start_index:end_index] = self.control.tolist()[0]
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