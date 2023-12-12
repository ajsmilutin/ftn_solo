import numpy as np
from scipy.interpolate import CubicSpline
import pinocchio as pin
from transitions import Machine
from robot_properties_solo.solo12wrapper import Solo12Robot
from ahrs.filters.madgwick import Madgwick
from utils.types import Plane, Trajectory


class Controller():
    states = ['start', 'step_fl', 'idle']
    step_dx = 0.1
    step_dy = 0.03
    step_dz = 0.02
    step_duration = 4.0

    def __init__(self):
        self.rest_pose = np.array(
            [0.0, 0.6, -1.57, 0.0, 0.6, -1.57, 0.0, 0.6, -1.57, 0.0, 0.6, -1.57], dtype=np.float64)
        self.control = None
        self.machine = Machine(model=self, states=Controller.states, initial='start')
        self.total_time = 4
        self.machine.add_transition(
            'tick', 'start', 'step_fl', conditions='go_to_start')
        self.machine.add_transition(
            'tick', 'step_fl', 'idle', conditions='do_step')
        self.machine.add_transition(
            'tick', 'idle', 'idle', conditions='do_nothing')
        self.machine.on_enter_step_fl(self.compute_step)
        self.madgwick = Madgwick()
        self.robot = Solo12Robot()
        self.old_t = 0
        self.dT = 0.001
        self.Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.eefs = [self.robot.fl_index, self.robot.fr_index,
                     self.robot.hl_index, self.robot.hr_index]
        self.shoulders = []
        for leg in ["FL", "FR", "HL", "HR"]:
            self.shoulders.append(self.robot.pin_robot.model.getFrameId(leg + "_UPPER_LEG"))
        self.foot_radius = 0.0175
        self.surface = None
        self.eef_trajectory = None
        self.sensor = None
        self.ref_pos = np.ndarray((3, len(self.eefs)), dtype=np.float64)
        self.init_pose()

    def estimate_pose(self, t, q, qv, sensor):
        touch_sensors = list(sensor["touch"].values())
        touch_transition = [False, False, False, False]
        self.full_dq = np.hstack((np.zeros((6), dtype=np.float64), qv))
        self.full_q = np.hstack((self.pos, self.Q[1:4], self.Q[0], q))
        self.robot.forward_robot(self.full_q, self.full_dq)
        if self.sensor is not None:
            old_values = list(self.sensor["touch"].values())
            touch_transition = touch_sensors and [not x for x in old_values]
        for index in range(len(self.eefs)):
            if touch_transition[index]:
                self.ref_pos[:, index] = self.robot.pin_robot.data.oMf[self.eefs[index]].translation
        self.sensor = sensor
        num_of_contacts = touch_sensors.count(True)
        if num_of_contacts <= 2:
            return
        ref_pos = np.zeros((num_of_contacts * 3, 1), dtype=np.float64)
        eef_pos = np.zeros((num_of_contacts * 3, 1), dtype=np.float64)
        full_J = np.zeros((3 * num_of_contacts, 3), dtype=np.float64)
        v = np.zeros((3, 1), dtype=np.float64)
        leg_num = 0
        for i in range(len(touch_sensors)):
            if touch_sensors[i]:
                ref_pos[leg_num * 3: (leg_num+1) * 3, 0] = self.ref_pos[:, i]
                leg_num += 1
        error = np.linalg.norm(ref_pos - eef_pos)
        old_error = 0
        delta_error = np.abs(error - old_error)
        while error > 0.0001 and delta_error > 0.000001:
            self.pos += 0.1 * v[:3, 0]
            self.full_q[:3] = self.pos
            self.robot.forward_robot(self.full_q, self.full_dq)
            leg_num = 0
            for i in range(len(touch_sensors)):
                if not touch_sensors[i]:
                    continue
                eef_pos[3 * leg_num: 3 * (leg_num + 1), 0] = self.robot.pin_robot.data.oMf[self.eefs[i]].translation
                full_J[3 * leg_num:3 * (leg_num + 1), :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                                            self.robot.pin_robot.data,
                                                                            self.eefs[i], pin.WORLD)[:3, :3]
                leg_num += 1
            old_error = error
            error = np.linalg.norm(ref_pos - eef_pos)
            delta_error = np.abs(error - old_error)
            v = np.linalg.pinv(full_J).dot(ref_pos - eef_pos)

    def init_pose(self):
        self.full_q = np.hstack((self.pos, self.Q[1:4], self.Q[0], np.zeros((12,), dtype=np.float64)))
        self.full_dq = np.zeros((18,), dtype=np.float64)
        self.robot.forward_robot(self.full_q, self.full_dq)
        pos = np.zeros((3,), dtype=np.float64)
        for index in self.eefs:
            pos = pos + self.robot.pin_robot.data.oMf[index].translation
        self.pos = -pos / len(self.eefs)
        self.pos[2] += self.foot_radius
        self.full_q[:3] = self.pos
        self.robot.forward_robot(self.full_q, self.full_dq)
        for index in range(len(self.eefs)):
            self.ref_pos[:, index] = self.robot.pin_robot.data.oMf[self.eefs[index]].translation

    def estimate_state(self, t, q, qv, sensor):
        self.dT = t - self.old_t
        self.old_t = t
        if (self.dT > 0):
            self.madgwick.Dt = self.dT
        self.Q = self.madgwick.updateIMU(
            self.Q, gyr=sensor["imu"][0], acc=sensor["imu"][1])
        self.estimate_pose(t, q, qv, sensor)

    def go_to_start(self, t, q, qv, sensor):
        self.control = min(t, self.total_time) / \
                       self.total_time * self.rest_pose
        return t > self.total_time

    def do_nothing(self, t, q, qv, sensor):
        pass

    def compute_step(self, t, q, qv, sensor):
        poses = np.zeros((3, len(self.eefs)), dtype=np.float64)
        for i, index in enumerate(self.eefs):
            poses[:, i] = self.robot.pin_robot.data.oMf[index].translation
        self.surface = Plane()
        self.surface.init_from_points(poses, np.asmatrix(self.pos).T)
        self.compute_eef_trajectory(t, self.eefs[0], self.shoulders[0], Controller.step_dx,
                                    Controller.step_dy, Controller.step_dz,
                                    Controller.step_duration)

    def compute_eef_trajectory(self, t, eef_ID, shoulder_ID, dx=0.07, dy=0.0, dz=0.04, step_duration=4.0):
        eef_pos = self.robot.pin_robot.data.oMf[eef_ID].translation
        end_ref = self.robot.pin_robot.data.oMf[shoulder_ID].translation
        start_pos = np.asarray(self.surface.transform_to_plane(np.asmatrix(eef_pos).T))
        end_pos = np.asarray(self.surface.transform_to_plane(np.asmatrix(end_ref).T)) + np.array([[dx], [dy], [0]])
        self.total_time = t + step_duration

        t_points = np.array([t, t + step_duration / 2, self.total_time], dtype=np.float64)
        x_points = np.array([start_pos[0][0], (start_pos[0][0] + end_pos[0][0]) / 2, end_pos[0][0]], dtype=np.float64)
        y_points = np.array([start_pos[1][0], (start_pos[1][0] + end_pos[1][0]) / 2, end_pos[1][0]], dtype=np.float64)
        z_points = np.array([start_pos[2][0], start_pos[2][0] + dz, start_pos[2][0]], dtype=np.float64)
        xyz_points = self.surface.transform_to_world(np.asarray([x_points, y_points, z_points]))
        self.eef_trajectory = Trajectory(t, self.total_time, eef_ID, CubicSpline(t_points, xyz_points.T))

    def do_step(self, t, q, qv, sensor):
        end_point = self.eef_trajectory.spline(t)
        eef_pose = self.robot.pin_robot.data.oMf[self.eef_trajectory.eef_ID].translation
        q_offset = 3 * self.eefs.index(self.eef_trajectory.eef_ID)
        v = np.zeros((3,), dtype=np.float64)
        self.full_q[7:] = q
        self.full_dq[6:] = qv
        old_error = 0
        error = np.linalg.norm(end_point.T - eef_pose.T)
        delta_error = np.abs(error - old_error)
        while error > 0.00001 and delta_error > 0.001:
            J = pin.getFrameJacobian(self.robot.pin_robot.model, self.robot.pin_robot.data,
                                     self.eef_trajectory.eef_ID, pin.WORLD)
            J = J[:3, 6 + q_offset:9 + q_offset]
            v = np.linalg.inv(J).dot(end_point.T - eef_pose.T)
            self.full_q[7 + q_offset:10 + q_offset] += v * self.dT
            self.robot.forward_robot(self.full_q, self.full_dq)
            eef_pose = self.robot.pin_robot.data.oMf[self.eef_trajectory.eef_ID].translation
            old_error = error
            error = np.linalg.norm(end_point.T - eef_pose.T)
            delta_error = np.abs(error - old_error)

        self.control[q_offset:q_offset + 3] += self.full_q[7 + q_offset:10 + q_offset] - q[q_offset:q_offset + 3]
        return t > self.eef_trajectory.end_time

    def compute_controll(self, t, q, qv, sensor):
        # put the controller here. This function is called inside the simulation.
        self.estimate_state(t, q, qv, sensor)
        self.tick(t, q, qv, sensor)
        return self.control
