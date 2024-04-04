import numpy as np
from transitions import Machine
from scipy.signal import chirp
from scipy.interpolate import CubicSpline
import csv

class ControllerIdent():
    states = ['start', 'move_knee', 'position_calf', 'move_hip', 'position_thigh','rotate_hip', 'return_to_start', 'idle']
    chirp_duration = 5.0
    chirp_F0 = 0.4
    chirp_F1 = 0.8
    prepare_duration = 2.0
    Kp = 8.0
    Kd = 0.1
    max_control = 1.8 # 0.025*8*9
    knee_torque = 0.25
    hip_torque = 0.25
    hip_rot_torque = 0.2

    def __init__(self, num_of_joints) -> None:
        self.machine = Machine(model = self, states = ControllerIdent.states, initial = 'start')
        self.machine.add_transition('tick', 'start', 'move_knee', conditions = 'prepare_start')
        self.machine.add_transition('tick', 'move_knee', 'position_calf', conditions = 'move_joint')
        self.machine.add_transition('tick', 'position_calf', 'move_hip', conditions = 'prepare_calf')
        self.machine.add_transition('tick', 'move_hip', 'position_thigh', conditions = 'move_joint')
        self.machine.add_transition('tick', 'position_thigh', 'rotate_hip', conditions='prepare_thigh')
        self.machine.add_transition('tick', 'rotate_hip', 'return_to_start', conditions = 'move_joint')
        self.machine.add_transition('tick', 'return_to_start', 'idle', conditions = 'go_to_start')
        self.machine.add_transition('tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_move_knee(self.prepare_move)
        self.machine.on_enter_move_hip(self.prepare_move)
        self.machine.on_enter_rotate_hip(self.prepare_move)
        self.machine.on_enter_position_calf(self.calculate_trajectory)
        self.machine.on_enter_position_thigh(self.calculate_trajectory)
        self.machine.on_enter_return_to_start(self.calculate_return_trajectory)
        self.machine.on_enter_idle(self.save_log)

        self.joints_num = num_of_joints

        self.control = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_position = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_velocity = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.B = 0*np.array([2.05205114e-02, 2.17915586e-02, 2.29703418e-02] * (self.joints_num // 3), dtype=np.float64)
        self.Fv = 0*np.array([8.81394325e-02, 8.67753849e-02, 1.18672339e-01] * (self.joints_num // 3), dtype=np.float64)
        self.transition_start = 0.0
        self.transition_end = 1.0
        self.dT = 0.001
        self.chirp_t = np.arange(0, self.chirp_duration + self.dT, self.dT / 2.0, dtype = np.float64)
        phase = 270
        self.chirp = chirp(self.chirp_t, self.chirp_F0, self.transition_end, self.chirp_F1, phi = phase)
        self.trajectory = CubicSpline

        self.debug_log = open("controller_log.txt", "w")
        self.log_rows = []

    def compute_control(self, t, q, qv):
        self.tick(t, q, qv)
        self.log_data(t, q, qv)
        return self.control

    def prepare_start(self, t, q, qv):
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        return t >= self.transition_end

    def prepare_move(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + self.chirp_duration


    def move_joint(self, t, q, qv):
        index = -1
        torque = 0.0
        control = 0.0
        scale = [1.0, 1.0, 1.0, 0.75, 1.0, 1.0, 0.75, 1.0, 1.0, 1.5, 1.0, 1.0]
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
            self.debug_log.write("Index out of range \n")
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        while (index < self.joints_num):
            self.control[index] = control * scale[index]
            index += 3
            if (self.joints_num == 13) and (index >=6 and index<= 8):
                index +=1
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def prepare_calf(self, t, q, qv):
        index = 2
        i = 0
        while (index < self.joints_num):
            self.ref_position[index] = self.trajectory(t)[i]
            self.ref_velocity[index] = self.trajectory(t, 1)[i]
            i += 1
            index += 3
            if (self.joints_num == 13) and (index >=6 and index <= 8):
                index += 1
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def prepare_thigh(self, t, q, qv):
        index = 1
        i = 0
        while (index < self.joints_num):
            self.ref_position[index] = self.trajectory(t)[i]
            self.ref_velocity[index] = self.trajectory(t, 1)[i]
            i += 1
            index += 3
            if (self.joints_num == 13) and (index >=6 and index <= 8):
                index += 1
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def calculate_trajectory(self, t, q, qv):
        self.ref_velocity = np.array([0.0] * self.joints_num, dtype=np.float64)
        end_position = -1
        i = 0
        index = -1
        columns_num = self.joints_num // 3
        q_points = np.ndarray((2, columns_num), dtype=np.float64)
        if self.machine.is_state('position_calf', self):
            end_position = 2.8
            index = 2
        elif self.machine.is_state('position_thigh', self):
            end_position = 0.5
            index = 1
        while (index < self.joints_num):
            q_points[:, i] = [q[index], end_position]
            i += 1
            index += 3
            if (self.joints_num == 13) and (index >=6 and index <= 8):
                index += 1

        self.transition_start = t
        self.transition_end = t + self.prepare_duration
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        self.trajectory = CubicSpline(t_points, q_points)

    def calculate_return_trajectory(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + 1.0
        t_points = np.array([t, self.transition_end], dtype=np.float64)
        q_points = np.ndarray((2, self.joints_num), dtype=np.float64)
        for i in range(self.joints_num):
            q_points[:, i] = [q[i], 0]
        self.trajectory = CubicSpline(t_points, q_points)

    def go_to_start(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def do_nothing(self, t, q, qv):
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return False
    
    def log_data(self, t, q, qv):
        row = [0.0] * (2 + 3 * self.joints_num)
        states = [self.states[1], self.states[3], self.states[5]]
        row[0] = 0.0
        for i in range(3):
            if self.machine.is_state(states[i], self):
                row[0] = float(i+1)
        if row[0] == 0.0:
            return
        row[1] = t
        start_index = 2
        end_index = self.joints_num + start_index
        row[start_index:end_index] = self.control.tolist()
        start_index = end_index
        end_index += self.joints_num
        row[start_index:end_index] = q.tolist()
        start_index = end_index
        end_index += self.joints_num
        row[start_index:end_index] = qv.tolist()
        self.log_rows.append(row)
        
    def save_log(self, t, q, qv):
        with open("ident_log.csv", "w") as log_file:
            log_csv = csv.writer(log_file)
            log_csv.writerows(self.log_rows)
            self.log_rows.clear()
        
