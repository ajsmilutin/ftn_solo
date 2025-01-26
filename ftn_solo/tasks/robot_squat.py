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

        self.splines = {
            "fl": {"arc": {}, "line": {}},
            "fr": {"arc": {}, "line": {}},
            "hl": {"arc": {}, "line": {}},
            "hr": {"arc": {}, "line": {}},
        }
        


        self.steps = []
        self.step = 0
        self.eps = 0.0018
        self.i = 0
        self.start = False
        self.backwards = False
        self.logger = logger
        self.R_y = np.eye(3)
        self.x_start = 0.196
        self.x_off = 0.05
        self.old_msg = Twist()
        self.define_splines(self.x_off)

    
    def define_movement(self,msg):
        if msg != self.old_msg:
            
            if abs(msg.linear.x) >= 0.1:
                msg.linear.x = 0.1
            elif abs(msg.linear.x) <= 0.02:
                msg.linear.x = 0.02


           
            if msg.linear.x <= 0:
                self.backwards = True
                self.start = True
            elif msg.linear.x == 0.02:
                self.start = False
               
            else:
                self.start = True
                self.backwards = False

            x_off = abs(msg.linear.x)
            self.define_splines(x_off,self.backwards)
            self.old_msg = msg
        else: 
            pass

    def define_splines(self,x_off,backwards = False):
        
        t_arc_points = np.array([0, 0.5, 1])
        x_front = np.array([self.x_start-x_off, self.x_start, self.x_start+x_off])
        z_arc = np.array([-0.25, -0.20, -0.25])
        x_back = np.array([self.x_start+x_off, self.x_start, self.x_start-x_off])
        z_line = np.array([-0.25, -0.25, -0.25])

        if backwards:
            x_f = x_back
            x_b = x_front
        else:
            x_f = x_front
            x_b = x_back

        for leg in self.splines.keys():

            if leg == "fl" or leg == "fr":
                x_arc = x_f
                x_line = x_b
            else:
                x_arc = -x_b
                x_line = -x_f

            self.splines[leg]["arc"]["x"] = CubicSpline(t_arc_points, x_arc)
            self.splines[leg]["arc"]["z"] = CubicSpline(t_arc_points, z_arc)
            self.splines[leg]["line"]["x"] = CubicSpline(t_arc_points, x_line)
            self.splines[leg]["line"]["z"] = CubicSpline(t_arc_points, z_line)


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

            s_t = 10 * (t_mod / T)**3 - 15 * (t_mod / T)**4 + 6 * (t_mod / T)**5
            s_dot = (30 * (t_mod / T)**2 - 60 * (t_mod / T)**3 + 30 * (t_mod / T)**4) / T
            s_ddot = (60 * (t_mod / T) - 180 * (t_mod / T) **
                    2 + 120 * (t / T)**3) / (T**2)

            motion = "arc" if leg in ["fr", "hl"] else "line"

        else:
            t_d = t_mod - T

            s_t = 10 * (t_d / T2)**3 - 15 * (t_d / T2)**4 + 6 * (t_d / T2)**5
            s_dot = (30 * (t_d / T2)**2 - 60 * (t_d / T2)**3 + 30 * (t_d / T2)**4) / T2
            s_ddot = (60 * (t_d / T2) - 180 * (t_d / T2) **
                    2 + 120 * (t_d / T2)**3) / (T2**2)

            motion = "line" if leg in ["fr", "hl"] else "arc"

        x_pos = self.splines[leg][motion]["x"](s_t)
        z_pos = self.splines[leg][motion]["z"](s_t)

        x_acc = self.splines[leg][motion]["x"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["x"](s_t, 1) * s_ddot
        z_acc = self.splines[leg][motion]["z"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["z"](s_t, 1) * s_ddot

        return np.array([x_pos, 0.1469 if "fl" in leg or "hl" in leg else -0.1469, z_pos]), \
            np.array([x_acc, 0, z_acc])
    


    def compute_control(self, t, position, velocity, sensors):

        leg_pos = []
        leg_acc = []
        

        
        
        if not self.start:
            tourques = self.joint_controller.rnea(
            self.steps, leg_acc, position, velocity, sensors['attitude'],t,0.5e6)
            return tourques
        
        else:

            for leg in ["fl", "fr", "hl", "hr"]:

                pos, acc = self.get_trajectory(t, leg,0.06,0.06)
                leg_pos.append(self.pin_robot.moveSE3(self.R_y, pos))
                leg_acc.append(acc)
    
            tourques = self.joint_controller.rnea(
                leg_pos, leg_acc, position, velocity, sensors['attitude'],t,2e6)
            
            return tourques

           

      
       

        
