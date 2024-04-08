import numpy as np
from transitions import Machine
from scipy.signal import chirp
from scipy.interpolate import CubicSpline


class ControllerTest():
    states = ['start', 'first_test', 'return_to_start_1', 'second_test',
              'return_to_start_2', 'third_test', 'return_to_start_3', 'idle']
    move_duration = 5.0
    SIN_W = 2 * 3.14 * 0.5  # f = 0.5Hz
    prepare_duration = 1.0
    Kp = 9.0  # 25
    Kd = 0.015  # 0.00725
    max_control = 1.8  # 0.025*8*9
    joint_sin_pos = np.array([0.5, 1.0, 1.57], dtype=np.float64)

    def __init__(self, num_of_joints, compensation) -> None:
        self.machine = Machine(
            model=self, states=ControllerTest.states, initial='start')
        self.machine.add_transition(
            'tick', 'start', 'first_test', conditions='prepare_start')
        self.machine.add_transition(
            'tick', 'first_test', 'return_to_start_1', conditions='move_all_joints')
        self.machine.add_transition(
            'tick', 'return_to_start_1', 'second_test', conditions='go_to_start')
        self.machine.add_transition(
            'tick', 'second_test', 'return_to_start_2', conditions='move_all_joints')
        self.machine.add_transition(
            'tick', 'return_to_start_2', 'third_test', conditions='go_to_start')
        self.machine.add_transition(
            'tick', 'third_test', 'return_to_start_3', conditions='move_all_joints')
        self.machine.add_transition(
            'tick', 'return_to_start_3', 'idle', conditions='go_to_start')
        self.machine.add_transition(
            'tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_first_test(self.prepare_move)
        self.machine.on_enter_second_test(self.prepare_move)
        self.machine.on_enter_third_test(self.prepare_move)
        self.machine.on_enter_return_to_start_1(
            self.calculate_return_trajectory)
        self.machine.on_enter_return_to_start_2(
            self.calculate_return_trajectory)
        self.machine.on_enter_return_to_start_3(
            self.calculate_return_trajectory)

        self.joints_num = num_of_joints

        self.control = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_position = np.array([0.0] * self.joints_num, dtype=np.float64)
        self.ref_velocity = np.array([0.0] * self.joints_num, dtype=np.float64)
        if compensation:
            self.B = np.array([2.05205114e-02, 2.17915586e-02, 2.29703418e-02]
                              * (self.joints_num // 3), dtype=np.float64)
            self.Fv = np.array([8.81394325e-02, 8.67753849e-02, 1.18672339e-01]
                               * (self.joints_num // 3), dtype=np.float64)
        else:
            self.B = np.zeros(self.joints_num, dtype=np.float64)
            self.Fv = np.zeros(self.joints_num, dtype=np.float64)
        self.transition_start = 0.0
        self.transition_end = 1.0
        self.dT = 0.001
        self.trajectory = CubicSpline

        self.log = open("controller_log.txt", "w")

    def compute_control(self, t, q, qv):
        self.tick(t, q, qv)
        return self.control

    def prepare_start(self, t, q, qv):
        self.control = self.ref_position
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
        position = np.sin(self.SIN_W * factor *
                          (t - self.transition_start)) * position
        velocity = np.cos(self.SIN_W * factor * (t -
                          self.transition_start)) * velocity * factor * self.SIN_W
        if self.joints_num == 13:
            self.ref_position[:6] = position
            self.ref_velocity[:6] = velocity
            self.ref_position[7:] = position
            self.ref_velocity[7:] = velocity
        else:
            self.ref_position = position
            self.ref_velocity = velocity
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
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
        position = np.sin(self.SIN_W * (t - self.transition_start)
                          ) * self.joint_sin_pos[index]
        velocity = np.cos(self.SIN_W * (t - self.transition_start)
                          ) * self.joint_sin_pos[index]
        while (index < self.joints_num):
            self.ref_position[index] = position
            self.ref_velocity[index] = velocity
            index += 3
            if (self.joints_num == 13) and (index >= 6 and index <= 8):
                index += 1
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def prepare_calf(self, t, q, qv):
        index = 2
        i = 0
        while (index < self.joints_num):
            self.ref_position[index] = self.trajectory(t)[i]
            self.ref_velocity[index] = self.trajectory(t, 1)[i]
            i += 1
            index += 3
            if (self.joints_num == 13) and (index >= 6 and index <= 8):
                index += 1
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def prepare_thigh(self, t, q, qv):
        index = 1
        i = 0
        while (index < self.joints_num):
            self.ref_position[index] = self.trajectory(t)[i]
            self.ref_velocity[index] = self.trajectory(t, 1)[i]
            i += 1
            index += 3
            if (self.joints_num == 13) and (index >= 6 and index <= 8):
                index += 1
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
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
            if (self.joints_num == 13) and (index >= 6 and index <= 8):
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
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (self.ref_velocity - qv) + \
            self.B * self.ref_velocity + self.Fv * np.sign(self.ref_velocity)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
        return t >= self.transition_end

    def do_nothing(self, t, q, qv):
        self.control = self.Kp * (self.ref_position - q) + self.Kd * (-qv)
        self.control = np.clip(
            self.control, -self.max_control, self.max_control)
        return False
