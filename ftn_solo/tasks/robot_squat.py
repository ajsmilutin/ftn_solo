import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm
from .task_base import TaskBase
from scipy.interpolate import CubicSpline
import time
from geometry_msgs.msg import Twist


class RobotMove(TaskBase):

    def __init__(self, num_joints, robot_version, config_yaml, logger, dt) -> None:
        super().__init__(num_joints, robot_version, config_yaml)
        self.pin_robot = PinocchioWrapper(robot_version, logger, dt)
        self.joint_controller = RneAlgorithm(
            num_joints, self.config["joint_controller"], robot_version, logger, dt)

        
        self.splines = {leg: {"arc": {}, "line": {}} for leg in ["fl", "fr", "hl", "hr"]}
     

        self.steps = []
        self.step = 0
        self.eps = 0.0018
        self.i = 0
        self.start = False
        self.backwards = False
        self.move_x = False
        self.move_y = False

        self.logger = logger
        self.R_y = np.eye(3)
        self.start_point = 0.1469
        self.off = 0.05
        self.old_msg = Twist()
        self.define_splines(self.off)

    def define_movement(self, msg):
        if msg != self.old_msg:
            
            if msg !=0:
                self.start = True

            if msg.linear.x !=0:
                self.backwards = msg.linear.x < 0
                self.move_x = True
                self.move_y = False
                self.start_point = 0.196
                off = np.clip(abs(msg.linear.x) , 0.02, 0.08)
            elif msg.linear.y !=0:
                self.backwards = msg.linear.y < 0
                self.move_y = True
                self.move_x = False
                self.start_point = 0.1469
                off = np.clip(abs(msg.linear.y) , 0.02, 0.04)
            elif msg.angular.z !=0:
                self.backwards = msg.angular.z < 0
                self.move_x = False
                self.move_y = False
                self.start_point = 0.1469     
                off = np.clip(abs(msg.angular.z) , 0.02, 0.08)
            else:               
                self.start=False
                off = 0



            self.define_splines(off)
            self.old_msg = msg
        else:
            pass

    def define_splines(self, off):

        t_arc_points = np.array([0, 0.5, 1])
        front = np.array(
            [ self.start_point-off,  self.start_point,  self.start_point+off])
        zarc = np.array([-0.25, -0.20, -0.25])
        back = np.array(
            [ self.start_point+off,  self.start_point,  self.start_point-off])
        zline = np.array([-0.25, -0.25, -0.25])

        x_f, x_b = (back, front) if self.backwards else (front, back)


        for leg in self.splines.keys():

            if self.move_x:
                if leg == "fl" or leg == "fr":
                    x_arc = x_f
                    x_line = x_b
                else:
                    x_arc = -x_b
                    x_line = -x_f
            elif self.move_y:
                if leg == "fr" or leg == "hr":
                    x_arc = -x_f
                    x_line = -x_b
                else:
                    x_arc = x_b
                    x_line = x_f
            else:
                if leg == "hr" or leg == "fl":
                    x_arc = x_f
                    x_line = x_b
                else:
                    x_arc = x_b
                    x_line = x_f

                if leg == "fr" or leg == "hr":
                    x_arc = -x_arc
                    x_line = -x_line

               
            self.splines[leg]["arc"]["x"] = CubicSpline(t_arc_points, x_arc)
            self.splines[leg]["arc"]["z"] = CubicSpline(t_arc_points, zarc)
            self.splines[leg]["line"]["x"] = CubicSpline(t_arc_points, x_line)
            self.splines[leg]["line"]["z"] = CubicSpline(t_arc_points, zline)

    def init_pose(self, q, dq):

        v1 = np.array([0.15, 0.20, -0.25])
        v2 = np.array([0.15, -0.20, -0.25])
        v3 = np.array([-0.15, 0.20, -0.25])
        v4 = np.array([-0.15, -0.20, -0.25])
        odmes1 = self.pin_robot.moveSE3(self.R_y, v1)
        odmes2 = self.pin_robot.moveSE3(self.R_y, v2)
        odmes3 = self.pin_robot.moveSE3(self.R_y, v3)
        odmes4 = self.pin_robot.moveSE3(self.R_y, v4)
        self.steps = [odmes1, odmes2, odmes3, odmes4]

        return self.steps

    def get_trajectory(self, t, leg, T, T2):

        T_total = T + T2
        t_mod = t % T_total

        if t_mod <= T:

            s_t = 10 * (t_mod / T)**3 - 15 * \
                (t_mod / T)**4 + 6 * (t_mod / T)**5
            s_dot = (30 * (t_mod / T)**2 - 60 * (t_mod / T)
                     ** 3 + 30 * (t_mod / T)**4) / T
            s_ddot = (60 * (t_mod / T) - 180 * (t_mod / T) **
                      2 + 120 * (t_mod / T)**3) / (T**2)

            motion = "arc" if leg in ["fr", "hl"] else "line"

        else:
            t_d = t_mod - T

            s_t = 10 * (t_d / T2)**3 - 15 * (t_d / T2)**4 + 6 * (t_d / T2)**5
            s_dot = (30 * (t_d / T2)**2 - 60 * (t_d / T2)
                     ** 3 + 30 * (t_d / T2)**4) / T2
            s_ddot = (60 * (t_d / T2) - 180 * (t_d / T2) **
                      2 + 120 * (t_d / T2)**3) / (T2**2)

            motion = "line" if leg in ["fr", "hl"] else "arc"

        x_pos = self.splines[leg][motion]["x"](s_t)
        z_pos = self.splines[leg][motion]["z"](s_t)

        x_acc = self.splines[leg][motion]["x"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["x"](s_t, 1) * s_ddot
        z_acc = self.splines[leg][motion]["z"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["z"](s_t, 1) * s_ddot

        if self.move_x:
            return np.array([x_pos, 0.1469 if "fl" in leg or "hl" in leg else -0.1469, z_pos]), \
                np.array([x_acc, 0, z_acc])
        else:
            return np.array([0.196 if "fl" in leg or "fr" in leg else -0.196, x_pos, z_pos]), \
                np.array([x_acc, 0, z_acc])

    def compute_control(self, t, position, velocity, sensors):

        leg_pos = []
        leg_acc = []

        if not self.start:
            tourques = self.joint_controller.rnea(
                self.steps, leg_acc, position, velocity, sensors['attitude'], t, 0.7e6)
            return tourques

        else:

            for leg in ["fl", "fr", "hl", "hr"]:

                pos, acc = self.get_trajectory(t, leg, 0.06, 0.06)
                leg_pos.append(self.pin_robot.moveSE3(self.R_y, pos))
                leg_acc.append(acc)

            tourques = self.joint_controller.rnea(
                leg_pos, leg_acc, position, velocity, sensors['attitude'], t, 2e6)

            return tourques
