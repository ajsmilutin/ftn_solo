import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm
from .task_base import TaskBase
from scipy.interpolate import CubicSpline
import time
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Imu
import rclpy


class RobotMove(TaskBase):

    def __init__(self, num_joints, robot_version, config_yaml, logger, dt, twist_pub=None, imu_pub=None) -> None:
        super().__init__(num_joints, robot_version, config_yaml)
        self.pin_robot = PinocchioWrapper(robot_version, logger, dt)
        self.joint_controller = RneAlgorithm(
            num_joints, self.config["joint_controller"], robot_version, logger, dt)
        
        # ROS publishers
        self.twist_publisher = twist_pub
        self.imu_publisher = imu_pub

        self.splines = {leg: {"arc": {"x": None, "y": None, "z": None}, 
                             "line": {"x": None, "y": None, "z": None}}
                        for leg in ["FL", "FR", "HL", "HR"]}

        self.steps = []
        self.step = 0
        self.eps = 0.0018
        self.i = 0
        self.start = False
        self.backwards = False
        self.move_x = False
        self.move_y = False
        self.rotate = False
        self.backwards_x = False
        self.backwards_y = False
        self.backwards_rot = False
        self.off_x = 0.0
        self.off_y = 0.0
        self.off_rot = 0.0
        self.off_time = time.time()

        self.ndq = np.zeros(18)
        self.nddq = np.zeros(18)

        self.logger = logger
        self.R_y = np.eye(3)
        self.off = 0.05
        self.old_msg = Twist()
        
        # Velocity ramping parameters for each axis
        self.target_off_x = 0.0
        self.target_off_y = 0.0
        self.target_off_rot = 0.0
        self.current_off_x = 0.0
        self.current_off_y = 0.0
        self.current_off_rot = 0.0
        
        # Separate ramp durations for acceleration and deceleration
        self.ramp_duration_accel = 1.5   # Time to accelerate (seconds)
        self.ramp_duration_decel = 2.5    # Time to decelerate (seconds) - can be longer for smoother stops
        
        self.ramp_start_time_x = 0.0
        self.ramp_start_time_y = 0.0
        self.ramp_start_time_rot = 0.0
        self.ramp_start_value_x = 0.0
        self.ramp_start_value_y = 0.0
        self.ramp_start_value_rot = 0.0
        self.ramping_x = False
        self.ramping_y = False
        self.ramping_rot = False
        
        # Gait timing parameters
        self.T_min = 0.04  # Minimum phase duration (high speed)
        self.T_max = 0.08  # Maximum phase duration (low speed)
        self.T = self.T_max  # Current phase duration
        
        # Phase tracking for smooth T transitions
        self.phase = 0.0  # Current phase [0, 1] within gait cycle
        self.last_t = 0.0  # Last time value
        
        # Track previous direction flags to detect changes
        self.prev_backwards_x = False
        self.prev_backwards_y = False
        self.prev_backwards_rot = False
        
        # Initialize splines at startup
        self.define_splines(0.005, 0.005, 0.005)
        
    def define_movement(self, msg):
        if msg != self.old_msg:

            self.start = True
            self.off = 0.05
            # Handle combined movements
            self.move_x = msg.linear.x != 0
            self.move_y = msg.linear.y != 0
            self.rotate = msg.angular.z != 0
            
            # Store old direction flags to detect changes
            old_backwards_x = self.backwards_x
            old_backwards_y = self.backwards_y
            old_backwards_rot = self.backwards_rot
            
            # Set direction flags - ONLY if actually moving in that direction
            self.backwards_x = msg.linear.x < 0 if msg.linear.x != 0 else self.backwards_x
            self.backwards_y = msg.linear.y > 0 if msg.linear.y != 0 else self.backwards_y
            self.backwards_rot = msg.angular.z < 0 if msg.angular.z != 0 else self.backwards_rot
            
            # For backwards compatibility
            self.backwards = self.backwards_x or self.backwards_y or self.backwards_rot
            
            # Set target offsets for each direction
            self.target_off_x = np.clip(abs(msg.linear.x), 0.0, 0.1) if self.move_x else 0.0
            self.target_off_y = np.clip(abs(msg.linear.y), 0.0, 0.1) if self.move_y else 0.0
            self.target_off_rot = np.clip(abs(msg.angular.z), 0.0, 0.1) if self.rotate else 0.0
            
            # Start ramping for X axis
            if self.target_off_x != self.current_off_x:
                self.ramp_start_value_x = self.current_off_x
                self.ramp_start_time_x = time.time()
                self.ramping_x = True
            
            # Start ramping for Y axis
            if self.target_off_y != self.current_off_y:
                self.ramp_start_value_y = self.current_off_y
                self.ramp_start_time_y = time.time()
                self.ramping_y = True
            
            # Start ramping for rotation
            if self.target_off_rot != self.current_off_rot:
                self.ramp_start_value_rot = self.current_off_rot
                self.ramp_start_time_rot = time.time()
                self.ramping_rot = True
            
            # If direction changed, immediately update splines
            if (self.backwards_x != old_backwards_x or 
                self.backwards_y != old_backwards_y or 
                self.backwards_rot != old_backwards_rot):
                self.logger.info(f"Direction changed! X:{old_backwards_x}→{self.backwards_x}, "
                               f"Y:{old_backwards_y}→{self.backwards_y}, "
                               f"Rot:{old_backwards_rot}→{self.backwards_rot}")
                self.define_splines(self.current_off_x, self.current_off_y, self.current_off_rot)
                self.prev_backwards_x = self.backwards_x
                self.prev_backwards_y = self.backwards_y
                self.prev_backwards_rot = self.backwards_rot
            
            if msg.linear.x == 0 and msg.linear.y == 0 and msg.angular.z == 0:
                self.off = 0.0
                self.off_time = time.time()
                
            self.old_msg = msg
    
    def update_velocity_ramp(self):
        """Linear ramp from current value to target over fixed duration for each axis"""
        splines_need_update = False
        
        # Ramp X axis
        if self.ramping_x:
            elapsed = time.time() - self.ramp_start_time_x
            # Use decel duration if slowing down, accel duration if speeding up
            is_decelerating = self.target_off_x < self.ramp_start_value_x
            duration = self.ramp_duration_decel if is_decelerating else self.ramp_duration_accel
            progress = min(elapsed / duration, 1.0)
            self.current_off_x = self.ramp_start_value_x + \
                                (self.target_off_x - self.ramp_start_value_x) * progress
            if progress >= 1.0:
                self.current_off_x = self.target_off_x
                self.ramping_x = False
                splines_need_update = True
        
        # Ramp Y axis
        if self.ramping_y:
            elapsed = time.time() - self.ramp_start_time_y
            is_decelerating = self.target_off_y < self.ramp_start_value_y
            duration = self.ramp_duration_decel if is_decelerating else self.ramp_duration_accel
            progress = min(elapsed / duration, 1.0)
            self.current_off_y = self.ramp_start_value_y + \
                                (self.target_off_y - self.ramp_start_value_y) * progress
            if progress >= 1.0:
                self.current_off_y = self.target_off_y
                self.ramping_y = False
                splines_need_update = True
        
        # Ramp rotation
        if self.ramping_rot:
            elapsed = time.time() - self.ramp_start_time_rot
            is_decelerating = self.target_off_rot < self.ramp_start_value_rot
            duration = self.ramp_duration_decel if is_decelerating else self.ramp_duration_accel
            progress = min(elapsed / duration, 1.0)
            self.current_off_rot = self.ramp_start_value_rot + \
                                  (self.target_off_rot - self.ramp_start_value_rot) * progress
            if progress >= 1.0:
                self.current_off_rot = self.target_off_rot
                self.ramping_rot = False
                splines_need_update = True
        
        # Check if direction flags changed
        direction_changed = (self.backwards_x != self.prev_backwards_x or
                           self.backwards_y != self.prev_backwards_y or
                           self.backwards_rot != self.prev_backwards_rot)
        
        # Only update splines when ramping completes, significant change, or direction changed
        # This prevents direction flag conflicts during deceleration
        if splines_need_update or direction_changed or \
           abs(self.current_off_x - self.off_x) > 0.005 or \
           abs(self.current_off_y - self.off_y) > 0.005 or \
           abs(self.current_off_rot - self.off_rot) > 0.005:
            self.off_x = self.current_off_x
            self.off_y = self.current_off_y
            self.off_rot = self.current_off_rot
            self.define_splines(self.off_x, self.off_y, self.off_rot)
            
            # Update previous direction flags
            self.prev_backwards_x = self.backwards_x
            self.prev_backwards_y = self.backwards_y
            self.prev_backwards_rot = self.backwards_rot
        
        # Update phase duration based on combined speed
        max_off = max(self.current_off_x / 0.03, self.current_off_y / 0.02, self.current_off_rot / 0.02)
        if max_off > 0.001:
            normalized_speed = np.clip(max_off, 0.0, 1.0)
            self.T = self.T_max - (self.T_max - self.T_min) * normalized_speed
        else:
            self.T = self.T_max

    def define_splines(self, off_x, off_y, off_rot):
        t_arc_points = np.array([0, 0.5, 1])
        
        # Use rotation for Y if rotating, otherwise use lateral
        if self.rotate:
            off_y = off_rot
        
        # X direction splines
        front_x = np.array([0.196-off_x, 0.196, 0.196+off_x])
        back_x = np.array([0.196+off_x, 0.196, 0.196-off_x])
        
        # Y direction splines
        front_y = np.array([0.1469-off_y, 0.1469, 0.1469+off_y])
        back_y = np.array([0.1469+off_y, 0.1469, 0.1469-off_y])
        
        # Z direction (height) splines
        zarc = np.array([-0.20, -0.15, -0.20])  # Arc has foot clearance
        zline = np.array([-0.20, -0.20, -0.20])  # Line stays on ground
        
        # Adjust direction based on backwards flags
        x_f, x_b = (back_x, front_x) if self.backwards_x else (front_x, back_x)
        y_f, y_b = (back_y, front_y) if self.backwards_y or self.backwards_rot else (front_y, back_y)
        
        for leg in self.splines.keys():
            # X movement splines
            if leg == "FL" or leg == "FR":  # Front legs
                x_arc = x_f
                x_line = x_b
            else:  # Hind legs
                x_arc = -x_b
                x_line = -x_f
        
            # Y movement pattern
            if leg == "FR" or leg == "HR":  # Right legs
                y_arc = -y_f
                y_line = -y_b
            else:  # Left legs
                y_arc = y_b
                y_line = y_f
            
            # Rotation movement adjustments
            if self.rotate:
                if leg == "HR" or leg == "FL":
                    y_arc = y_f
                    y_line = y_b
                else:
                    y_arc = y_b
                    y_line = y_f

                if leg == "FR" or leg == "HR":
                    y_arc = -y_arc
                    y_line = -y_line 
            
            # Create splines for all directions
            self.splines[leg]["arc"]["x"] = CubicSpline(t_arc_points, x_arc)
            self.splines[leg]["arc"]["y"] = CubicSpline(t_arc_points, y_arc)
            self.splines[leg]["arc"]["z"] = CubicSpline(t_arc_points, zarc)
            
            self.splines[leg]["line"]["x"] = CubicSpline(t_arc_points, x_line)
            self.splines[leg]["line"]["y"] = CubicSpline(t_arc_points, y_line)
            self.splines[leg]["line"]["z"] = CubicSpline(t_arc_points, zline)

    def init_pose(self, q, dq):

        v1 = np.array([0.18, 0.15, -0.25])
        v2 = np.array([0.18, -0.15, -0.25])
        v3 = np.array([-0.18, 0.15, -0.25])
        v4 = np.array([-0.18, -0.15, -0.25])
        odmes1 = self.pin_robot.moveSE3(self.R_y, v1)
        odmes2 = self.pin_robot.moveSE3(self.R_y, v2)
        odmes3 = self.pin_robot.moveSE3(self.R_y, v3)
        odmes4 = self.pin_robot.moveSE3(self.R_y, v4)
        self.steps = [odmes1, odmes2, odmes3, odmes4]

        return self.steps

    def recovery_pose(self):
        v1 = np.array([0.196, 0.1469, -0.14])
        v2 = np.array([0.196, -0.1469, -0.36])
        v3 = np.array([-0.196, 0.1469, -0.36])
        v4 = np.array([-0.196, -0.1469, -0.36])
        odmes1 = self.pin_robot.moveSE3(self.R_y, v1)
        odmes2 = self.pin_robot.moveSE3(self.R_y, v2)
        odmes3 = self.pin_robot.moveSE3(self.R_y, v3)
        odmes4 = self.pin_robot.moveSE3(self.R_y, v4)
        self.steps = [odmes1, odmes2, odmes3, odmes4]

        return self.steps
        

    def get_trajectory(self, t, leg, T, T2):
        # Use phase-based timing instead of absolute time
        # phase is in [0, 1] for full gait cycle
        # First half (phase < 0.5) is phase 1, second half is phase 2
        
        if self.phase < 0.5:
            # Phase 1: use first half of cycle
            tau = self.phase * 2.0  # Map [0, 0.5] to [0, 1]
            t_mod = tau * T  # Equivalent time in phase 1

            s_t = 10 * (t_mod / T)**3 - 15 * \
                (t_mod / T)**4 + 6 * (t_mod / T)**5
            s_dot = (30 * (t_mod / T)**2 - 60 * (t_mod / T)
                     ** 3 + 30 * (t_mod / T)**4) / T
            s_ddot = (60 * (t_mod / T) - 180 * (t_mod / T) **
                      2 + 120 * (t_mod / T)**3) / (T**2)

            motion = "arc" if leg in ["FR", "HL"] else "line"

        else:
            # Phase 2: use second half of cycle
            tau = (self.phase - 0.5) * 2.0  # Map [0.5, 1] to [0, 1]
            t_d = tau * T2  # Equivalent time in phase 2

            s_t = 10 * (t_d / T2)**3 - 15 * (t_d / T2)**4 + 6 * (t_d / T2)**5
            s_dot = (30 * (t_d / T2)**2 - 60 * (t_d / T2)
                     ** 3 + 30 * (t_d / T2)**4) / T2
            s_ddot = (60 * (t_d / T2) - 180 * (t_d / T2) **
                      2 + 120 * (t_d / T2)**3) / (T2**2)

            motion = "line" if leg in ["FR", "HL"] else "arc"

        x_pos = self.splines[leg][motion]["x"](s_t)
        z_pos = self.splines[leg][motion]["z"](s_t)

        x_vel = self.splines[leg][motion]["x"](s_t, 1) * s_dot
        z_vel = self.splines[leg][motion]["z"](s_t, 1) * s_dot

        x_acc = self.splines[leg][motion]["x"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["x"](s_t, 1) * s_ddot
        z_acc = self.splines[leg][motion]["z"](
            s_t, 2) * (s_dot**2) + self.splines[leg][motion]["z"](s_t, 1) * s_ddot

        # Get Y position, velocity and acceleration from splines
        y_pos = self.splines[leg][motion]["y"](s_t)
        y_vel = self.splines[leg][motion]["y"](s_t, 1) * s_dot
        y_acc = self.splines[leg][motion]["y"](s_t, 2) * (s_dot**2) + self.splines[leg][motion]["y"](s_t, 1) * s_ddot
        
        # Return full 3D trajectory with both X and Y components
        return np.array([x_pos, y_pos, z_pos]), \
               np.array([x_vel, y_vel, z_vel]), \
               np.array([x_acc, y_acc, z_acc])

    def compute_control(self, t, position, velocity, sensors):
        # Update velocity ramping only if actively ramping
        if self.ramping_x or self.ramping_y or self.ramping_rot:
            self.update_velocity_ramp()
        
        # Update phase based on time delta and current T
        if self.start:
            dt = t - self.last_t
            if dt > 0:  # Avoid issues on first call
                # Increment phase based on current gait frequency
                T_total = 2.0 * self.T
                phase_increment = dt / T_total
                self.phase = (self.phase + phase_increment) % 1.0  # Keep in [0, 1]
            self.last_t = t
        
        if self.off==0:
            if time.time() - self.off_time > 3:
                # self.recovery_pose()
                self.start = False
        
        self.ndq.fill(0)
        self.nddq.fill(0)
        # self.logger.info("Current ndq: {}".format(self.nddq))
        self.joint_controller.calculate_kinematics(position, velocity)
        if not self.start:
            for x, leg in enumerate(["FL", "FR", "HL", "HR"]):
                dq,ddq = self.joint_controller.calculate_acceleration(leg,self.steps[x],np.zeros(3),np.zeros(3))
                self.ndq += dq
                self.nddq += ddq

            tourques = self.joint_controller.get_tourqe(self.ndq,self.nddq)
            # self.logger.info("Sovle time: {}".format(time.time() - start_time))
            return tourques

        else:

            for leg in ["FL", "FR", "HL", "HR"]:
                pos, vel,acc = self.get_trajectory(t, leg, self.T, self.T)
                ref_pos = self.joint_controller.moveSE3(self.R_y, pos)
                dq,ddq = self.joint_controller.calculate_acceleration(leg,ref_pos,vel,acc)
                self.ndq += dq
                self.nddq += ddq

            tourques = self.joint_controller.get_tourqe(self.ndq,self.nddq)
            return tourques
    
   