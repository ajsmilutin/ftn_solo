from operator import indexOf
import numpy as np
from transitions import Machine
from scipy.signal import chirp
from scipy.interpolate import CubicSpline

class ControllerIdent():
    states = ['start', 'move_knee', 'position_calf', 'move_hip', 'position_thigh','rotate_hip', 'return_to_start', 'idle']
    chirp_duration = 5.0
    chirp_F0 = 0.4
    chirp_F1 = 0.8
    prepare_duration = 1.0
    Kp = 25.0 #25
    Kd = 0.00725 #0.00725
    max_control = 1.8 # 0.025*8*9
    knee_torque = 0.057
    hip_torque = 0.032
    hip_rot_torque = 0.018
    
    def __init__(self) -> None:
        self.machine = Machine(model = self, states = ControllerIdent.states, initial = 'start')
        self.machine.add_transition('tick', 'start', 'move_knee', conditions = 'prepare_start')
        self.machine.add_transition('tick', 'move_knee', 'position_calf', conditions = 'move_joint')
        self.machine.add_transition('tick', 'position_calf', 'move_hip', conditions = 'prepare_calf')
        self.machine.add_transition('tick', 'move_hip', 'position_thigh', conditions = 'move_joint')
        self.machine.add_transition('tick', 'position_thigh', 'rotate_hip', conditions='prepare_thigh')
        self.machine.add_transition('tick', 'rotate_hip', 'return_to_start', conditions = 'move_joint')
        self.machine.add_transition('tick', 'return_to_start', 'idle', conditions = 'go_to_start')
        self.machine.add_transition('tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_move_knee(self.calculate_chirp)
        self.machine.on_enter_move_hip(self.calculate_chirp)
        self.machine.on_enter_rotate_hip(self.calculate_chirp)
        self.machine.on_enter_position_calf(self.calculate_trajectory)
        self.machine.on_enter_position_thigh(self.calculate_trajectory)
        self.machine.on_enter_return_to_start(self.calculate_return_trajectory)
        
        self.control = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.ref_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.transition_start = 0.0
        self.transition_end = 1.0
        self.dT = 0.001
        self.chirp_t = np.arange(0, self.chirp_duration + self.dT, self.dT / 2.0, dtype = np.float64)
        phase = 270
        self.chirp = chirp(self.chirp_t, self.chirp_F0, self.transition_end, self.chirp_F1, phi = phase)
        self.trajectory = CubicSpline
        
        self.log = open("controller_log.txt", "w")
        
    def compute_control(self, t, q, qv):
        self.tick(t, q, qv)
        return self.control
    
    def prepare_start(self, t, q, qv):
        self.control = self.ref_position
        return t >= self.transition_end
    
    def calculate_chirp(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + self.chirp_duration

        
    def move_joint(self, t, q, qv):
        index = -1
        torque = 0.0
        control = 0.0
        if self.machine.is_state('move_knee', self):
            index = 2
            torque = self.knee_torque
        elif self.machine.is_state('move_hip', self):
            index = 1
            torque = self.hip_torque
        elif self.machine.is_state('rotate_hip', self):
            index = 0
            torque = self.hip_rot_torque
        try:
            chirp_index = np.where(self.chirp_t >= t - self.transition_start)[0][0]
            control = self.chirp[chirp_index] * torque
        except:
            self.log.write("Index out of range \n")
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control[index] = control
        self.control[index + 3] = control
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def prepare_calf(self, t, q, qv):
        self.ref_position[2] = self.trajectory(t)[0]
        self.ref_position[5] = self.trajectory(t)[1]
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def prepare_thigh(self, t, q, qv):
        self.ref_position[1] = self.trajectory(t)[0]
        self.ref_position[4] = self.trajectory(t)[1]
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def calculate_trajectory(self, t, q, qv):
        start_position = -1
        end_position = -1
        q_points = np.ndarray((2, 2), dtype=np.float64)
        if self.machine.is_state('position_calf', self):
            end_position = 2.8
            q_points[:, 0] = [q[2], end_position] 
            q_points[:, 1] = [q[5], end_position] 
        elif self.machine.is_state('position_thigh', self):
            end_position = 1.45
            q_points[:, 0] = [q[1], end_position] 
            q_points[:, 1] = [q[4], end_position] 
        self.transition_start = t
        self.transition_end = t + self.prepare_duration
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        self.trajectory = CubicSpline(t_points, q_points)
        
    def calculate_return_trajectory(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + 1.0
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        q_points = np.ndarray((2, 6), dtype=np.float64)
        for i in range(6):
            q_points[:, i] = [q[i], 0]
        self.trajectory = CubicSpline(t_points, q_points)
    
    def go_to_start(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def do_nothing(self, t, q, qv):
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return False
