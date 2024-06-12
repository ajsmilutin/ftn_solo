import numpy as np
from transitions import Machine
from scipy.signal import chirp
from scipy.interpolate import CubicSpline
import csv

class ControllerTest():
    states = ['start', 'first_test', 'return_to_start_1', 'second_test', 'return_to_start_2','third_test', 'return_to_start_3', 'idle']
    move_duration = 5.0
    SIN_W = 2 * 3.14 * 0.5 # f = 0.5Hz
    prepare_duration = 1.0
    Kp = 2.0 
    Kd = 0.0125
    max_control = 1.8 # 0.025*8*9
    joint_sin_pos = np.array([0.5, 1.0, 1.57], dtype=np.float64)
    friction_velocity_cutoff = 0.1
    
    def __init__(self, num_of_joints, compensation) -> None:
        self.machine = Machine(model = self, states = ControllerTest.states, initial = 'start')
        self.machine.add_transition('tick', 'start', 'first_test', conditions = 'prepare_start')
        self.machine.add_transition('tick', 'first_test', 'return_to_start_1', conditions = 'move_all_joints')
        self.machine.add_transition('tick', 'return_to_start_1', 'second_test', conditions = 'go_to_start')
        self.machine.add_transition('tick', 'second_test', 'return_to_start_2', conditions = 'move_all_joints')
        self.machine.add_transition('tick', 'return_to_start_2', 'third_test', conditions='go_to_start')
        self.machine.add_transition('tick', 'third_test', 'return_to_start_3', conditions = 'move_all_joints')
        self.machine.add_transition('tick', 'return_to_start_3', 'idle', conditions = 'go_to_start')
        self.machine.add_transition('tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_first_test(self.prepare_move)
        self.machine.on_enter_second_test(self.prepare_move)
        self.machine.on_enter_third_test(self.prepare_move)
        self.machine.on_enter_return_to_start_1(self.calculate_return_trajectory)
        self.machine.on_enter_return_to_start_2(self.calculate_return_trajectory)
        self.machine.on_enter_return_to_start_3(self.calculate_return_trajectory)
        self.machine.on_enter_idle(self.save_log)
        
        self.joints_num = num_of_joints
        
        self.control = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_position = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_velocity = np.array([0.0] * self.joints_num, dtype=np.float64)
        if compensation:
            self.B = np.array([0.02545695, 0.01706673, 0.01926639, 0.02819381, 0.02029258, 0.01606315, 0.01687575, 0.01313175, 0.01549928, 0.02191509, 0.01881964,0.01744183 ], dtype=np.float64)
            self.Fv = np.array([0.06743854, 0.09737804, 0.1287155, 0.04454953, 0.12968497, 0.09906573, 0.05082791, 0.09970842, 0.09362089, 0.15870984, 0.08582857, 0.08365718], dtype=np.float64)
        else:
            self.B = np.zeros(self.joints_num, dtype=np.float64)
            self.Fv = np.zeros(self.joints_num, dtype=np.float64)
        self.transition_start = 0.0
        self.transition_end = 1.0
        self.dT = 0.001
        self.trajectory = CubicSpline
        
        self.log_rows = []
        
    def compute_control(self, t, q, qv):
        self.tick(t, q, qv)
        self.log_data(t, q, qv)
        return self.control
    
    def prepare_start(self, t, q, qv):
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv) + self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def prepare_move(self, t, q, qv):
        self.transition_start = t
        self.transition_end = t + self.move_duration
        self.ref_velocity = np.zeros(self.joints_num, dtype=np.float64)
        
    def move_all_joints(self, t, q, qv):
        if self.machine.is_state('first_test', self):
            factor = 1.0
        elif self.machine.is_state('second_test', self):
            factor = 0.8
        elif self.machine.is_state('third_test', self):
            factor = 0.4
        position = np.tile(self.joint_sin_pos, (self.joints_num // 3))
        velocity = np.tile(self.joint_sin_pos, (self.joints_num // 3))
        position = np.sin(self.SIN_W * factor * (t - self.transition_start)) * position
        velocity = np.cos(self.SIN_W * factor * (t - self.transition_start)) * velocity * factor * self.SIN_W
        if self.joints_num == 13:
            self.ref_position[:6] = position
            self.ref_velocity[:6] = velocity
            self.ref_position[7:] = position
            self.ref_velocity[7:] = velocity
        else:
            self.ref_position = position
            self.ref_velocity = velocity
        friction_velocity = np.where(abs(self.ref_velocity) > self.friction_velocity_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(friction_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end


    def move_joint(self, t, q, qv):
        index = -1
        position = 0.0
        velocity = 0.0
        if self.machine.is_state('first_test', self):
            index = 2
        elif self.machine.is_state('second_test', self):
            index = 1
        elif self.machine.is_state('third_test', self):
            index = 0
        position = np.sin(self.SIN_W * (t - self.transition_start)) * self.joint_sin_pos[index]
        velocity = np.cos(self.SIN_W * (t - self.transition_start)) * self.joint_sin_pos[index] * self.SIN_W
        while (index < self.joints_num):
            self.ref_position[index] = position
            self.ref_velocity[index] = velocity
            index += 3
            if (self.joints_num == 13) and (index >=6 and index<= 8):
                index +=1
        friction_velocity = np.where(abs(self.ref_velocity) > self.friction_velocity_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity  + self.Fv * np.sign(friction_velocity)
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
        friction_velocity = np.where(abs(self.ref_velocity) > self.friction_velocity_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(friction_velocity)
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
        friction_velocity = np.where(abs(self.ref_velocity) > self.friction_velocity_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(friction_velocity)
        self.control = np.clip(self.control, -self.max_control, self.max_control)
        return t >= self.transition_end
    
    def calculate_trajectory(self, t, q, qv):
        self.ref_velocity = np.zeros(self.joints_num, dtype=np.float64)
        end_position = -1
        i = 0
        index = -1
        columns_num = self.joints_num // 3
        q_points = np.ndarray((2, columns_num), dtype=np.float64)
        if self.machine.is_state('return_to_start_1', self):
            end_position = 0
            index = 2
        elif self.machine.is_state('return_to_start_2', self):
            end_position = 0
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
        friction_velocity = np.where(abs(self.ref_velocity) > self.friction_velocity_cutoff, self.ref_velocity, 0)
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + self.B * self.ref_velocity + self.Fv * np.sign(friction_velocity)
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
        with open("test_log.csv", "w") as log_file:
            log_csv = csv.writer(log_file)
            log_csv.writerows(self.log_rows)
            self.log_rows.clear()
