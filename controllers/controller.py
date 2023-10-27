import numpy as np
from transitions import Machine
from robot_properties_solo.solo12wrapper import Solo12Robot
from ahrs.filters.madgwick import Madgwick


class Controller():
    states = ['start', 'step_fl', 'idle']

    def __init__(self):
        self.rest_pose = np.array(
            [0.0, 0.825, -1.57, 0.0, -0.825, 1.57, 0.0, 0.825, -1.57, 0.0, -0.825, 1.57], dtype=np.float64)
        self.machine = Machine(
            model=self, states=Controller.states, initial='start')
        self.total_time = 10
        self.machine.add_transition(
            'tick', 'start', 'step_fl', conditions='go_to_start', before='reset_pose')
        self.machine.add_transition(
            'tick', 'step_fl', 'idle', conditions='do_step')
        self.machine.add_transition(
            'tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_step_fl(self.compute_step)
        self.madgwick = Madgwick()
        self.robot = Solo12Robot()
        self.old_t = 0
        self.dT = 0.001
        self.Q = np.array([1.0,  0.0,  0.0,  0.0], dtype=np.float64)
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.eefs = [self.robot.fl_index, self.robot.fr_index,
                   self.robot.hl_index, self.robot.hr_index]

    def reset_pose(self, t, q, qv, sensor):
        self.robot.forward_robot(self.full_q, self.full_dq)
        pos = np.zeros((3,), dtype=np.float64)
        for index in self.eefs:
            pos = pos + self.robot.pin_robot.data.oMf[index].translation
        self.pos = -pos/len(self.eefs)

    def estimate_state(self, t, q, qv, sensor):
        self.dT = t-self.old_t
        self.old_t = t
        if (self.dT > 0):
            self.madgwick.Dt = self.dT
        self.Q = self.madgwick.updateIMU(
            self.Q, gyr=sensor["imu"][0], acc=sensor["imu"][1])
        self.full_q = np.hstack((self.pos, self.Q[1:4], self.Q[0], q))
        self.full_dq = np.hstack((np.zeros((6), dtype=np.float64), qv))

    def go_to_start(self, t, q, qv, sensor):
        self.control = min(t, self.total_time) / \
            self.total_time * self.rest_pose
        return t > self.total_time

    def do_nothing(self, t, q, qv, sensor):
        pass

    def compute_step(self, t, q, qv, sensor):
        poses = np.zeros((3, len(self.eefs)), dtype=np.float64)
        for i, index in enumerate(self.eefs):
            poses[:,i] = self.robot.pin_robot.data.oMf[index].translation


    def do_step(self, t, q, qv, sensor):
        return self.do_nothing(t, q, qv, sensor)

    def compute_controll(self, t, q, qv, sensor):
        # put the controller here. This function is called inside the simulation.
        self.estimate_state(t,  q, qv, sensor)
        self.tick(t, q, qv, sensor)
        return self.control
