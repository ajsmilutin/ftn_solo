from operator import indexOf
import numpy as np
from transitions import Machine
from scipy.signal import chirp
from scipy.interpolate import CubicSpline

class ControllerIdent():
    states = ['start', 'move_knee', 'position_calf', 'move_hip', 'position_thigh','rotate_hip', 'return_to_start', 'idle']
    chirp_duration = 5.0
    chirp_F0 = 0.5
    chirp_F1 = 3.0
    prepare_duration = 1.0
    
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
        self.transition_end = 1.0
        self.transition_start = 0.0
        self.dT = 0.001
        self.chirp_t = None
        self.chirp = None
        self.trajectory = CubicSpline
        
        self.log = open("controller_log.txt", "w")
        
    def compute_control(self, t, q, qv):
        self.tick(t, q, qv)
        return self.control
    
    def prepare_start(self, t, q, qv):
        self.control = 25 * (self.ref_position - q) + 0.00725 * (self.ref_position - qv)
        return t >= self.transition_end
    
    def calculate_chirp(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + self.chirp_duration
        self.chirp_t = np.arange(0, self.chirp_duration + self.dT, self.dT / 2.0, dtype = np.float64)
        phase = int(90 - 360 * self.chirp_F0)
        self.chirp = chirp(self.chirp_t, self.chirp_F0, self.transition_end, self.chirp_F1, phi = phase)
        
    def move_joint(self, t, q, qv):
        index = -1
        max_angle = -1
        if self.machine.is_state('move_knee', self):
            index = 2
            max_angle = 1.57
        elif self.machine.is_state('move_hip', self):
            index = 1
            max_angle = 1.57
        elif self.machine.is_state('rotate_hip', self):
            max_angle = 0.785
            index = 0
        try:
            chirp_index = np.where(self.chirp_t >= t - self.transition_start)[0][0]
            #self.log.write(str(self.chirp_t[index] - t))
            #self.log.write("\n")
            self.ref_position[index] = self.chirp[chirp_index] * max_angle
        except:
            self.log.write("Index out of range \n")
        self.control = 25 * (self.ref_position - q) + 0.00725 * (-qv)
        return t >= self.transition_end
    
    def prepare_calf(self, t, q, qv):
        self.ref_position[2] = self.trajectory(t)
        self.control = 25 * (self.ref_position - q) + 0.00725 * (-qv)
        return t >= self.transition_end
    
    def prepare_thigh(self, t, q, qv):
        self.ref_position[1] = self.trajectory(t)
        self.control = 25 * (self.ref_position - q) + 0.00725 * (-qv)
        return t >= self.transition_end
    
    def calculate_trajectory(self, t, q, qv):
        start_position = -1
        end_position = -1
        if self.machine.is_state('position_calf', self):
            start_position = q[2]
            end_position = 3.14
        elif self.machine.is_state('position_thigh', self):
            start_position = q[1]
            end_position = 1.57
        self.transition_start = t
        self.transition_end = t + self.prepare_duration
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        q_points = np.array([start_position, end_position], dtype=np.float64)
        self.trajectory = CubicSpline(t_points, q_points)
        
    def calculate_return_trajectory(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + 1.0
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        q_points = np.ndarray((2, 3), dtype=np.float64)
        for i in range(3):
            q_points[:, i] = [q[i], 0]
        self.trajectory = CubicSpline(t_points, q_points)
    
    def go_to_start(self, t, q, qv):
        self.ref_position[:3] = self.trajectory(t)
        self.control = 25 * (self.ref_position - q) + 0.00725 * (-qv)
        return t >= self.transition_end
    
    def do_nothing(self, t, q, qv):
        self.control = 25 * (self.ref_position - q) + 0.00725 * (self.ref_position - qv)
        return False
