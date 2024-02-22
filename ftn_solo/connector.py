#!/usr/bin/env python
import sys
from tracemalloc import start
import mujoco
import mujoco.viewer
import pybullet
import rclpy
from ftn_solo.utils.bullet_env import BulletEnvWithGround
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from rosgraph_msgs.msg import Clock
import time
import math
import yaml
import csv
from robot_properties_solo.robot_resources import Resources
from ftn_solo.controllers.controller_ident import ControllerIdent


def RPY2Quat(rpy):
    q1 = np.ndarray((4,), dtype=np.float64)
    q2 = np.ndarray((4,), dtype=np.float64)
    q3 = np.ndarray((4,), dtype=np.float64)
    mujoco.mju_axisAngle2Quat(q1, [0, 0, 1], rpy[2])
    mujoco.mju_axisAngle2Quat(q2, [0, 1, 0], rpy[1])
    mujoco.mju_mulQuat(q3, q1, q2)
    mujoco.mju_axisAngle2Quat(q2, [1, 0, 0], rpy[0])
    mujoco.mju_mulQuat(q1, q3, q2)
    return q1


class Connector():
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        self.resources = Resources(robot_version)
        self.logger = logger


class RobotConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)
        import libodri_control_interface_pywrap as oci
        self.running = True
        with open(self.resources.config_path, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                self.joint_names = data["robot"]["joint_modules"]["joint_names"]
            except Exception as exc:
                raise exc
        self.robot = oci.robot_from_yaml_file(self.resources.config_path)
        self.robot.initialize(np.array([0]*self.robot.joints.number_motors))
        self.running = True
        self.controller = ControllerIdent(self.robot.joints.number_motors)

    def get_data(self):
        self.robot.parse_sensor_data()
        return self.robot.joints.positions, self.robot.joints.velocities

    def set_torques(self, torques):
        self.robot.joints.set_torques(torques)

    def is_running(self):
        return self.running

    def step(self):
        self.robot.send_command_and_wait_end_of_cycle(0.001)
        return True


class PybulletConnector(Connector):
    def __init__(self, robot_version, logger, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0], *args, **kwargs) -> None:
        super().__init__(robot_version, logger)

        self.env = BulletEnvWithGround(robot_version)
        orn = pybullet.getQuaternionFromEuler(rpy)
        self.nanoseconds = int(self.env.dt*1e9)
        self.logger = logger
        self.robot_id = pybullet.loadURDF(
            self.resources.urdf_path,
            pos,
            orn,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=fixed,
        )

        self.joint_names = []
        self.joint_ids = []
        self.end_effector_ids = []
        self.running = True

        for ji in range(pybullet.getNumJoints(self.robot_id)):
            if pybullet.JOINT_FIXED != pybullet.getJointInfo(self.robot_id, ji)[2]:
                self.joint_names.append(pybullet.getJointInfo(
                    self.robot_id, ji)[1].decode("UTF-8"))
                self.joint_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0])
            else:
                self.end_effector_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0])

        pybullet.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros(len(self.joint_ids)),
        )

    def get_data(self):
        q = np.empty(len(self.joint_ids))
        dq = np.empty(len(self.joint_ids))

        joint_states = pybullet.getJointStates(self.robot_id, self.joint_ids)

        for i in range(len(self.joint_ids)):
            q[i] = joint_states[i][0]
            dq[i] = joint_states[i][1]

        return q, dq

    def set_torques(self, torques):
        pybullet.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            pybullet.TORQUE_CONTROL,
            forces=torques
        )

    def step(self):
        self.env.step(True)
        return True

    def is_running(self):
        return self.running


class MujocoConnector(Connector):
    def __init__(self, robot_version, logger, use_gui=True, start_paused=False, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0], *args, **kwargs) -> None:
        super().__init__(robot_version, logger)
        self.model = mujoco.MjModel.from_xml_path(self.resources.mjcf_path)
        self.model.opt.timestep = 1e-3
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = pos
        logger.error(str(rpy))
        self.data.qpos[3:7] = RPY2Quat(rpy)
        logger.error(str(self.data.qpos))
        self.data.qpos[7:] = 0
        self.data.qvel[:] = 0
        if fixed:
            self.model.body("base_link").jntnum = 0
        self.joint_names = [self.model.joint(
            i+1).name for i in range(self.model.nu)]
        self.paused = start_paused
        self.use_gui = use_gui
        self.viewer = None
        self.running = True
        self.nanoseconds = int(self.model.opt.timestep*1e9)
        self.controller = ControllerIdent()
        self.controller.dT = self.model.opt.timestep
        if self.use_gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_right_ui=False, key_callback=self.key_callback)

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused
        elif keycode == 256:  # ESC
            self.running = False

    def get_data(self):
        return self.data.qpos[7:], self.data.qvel[6:]

    def set_torques(self, torques):
        self.data.ctrl = torques

    def is_running(self):
        return self.running

    def step(self):
        if self.paused:
            time.sleep(self.model.opt.timestep)
            return False
        step_start = time.time()
        mujoco.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        time_until_next_step = self.model.opt.timestep - \
            (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        return True


class ConnectorNode(Node):
    def __init__(self):
        super().__init__("first_node")
        self.declare_parameter('hardware', rclpy.Parameter.Type.STRING)
        hardware = self.get_parameter('hardware').get_parameter_value().string_value
        self.time_publisher = None
        if hardware.lower() != "robot":
            self.time_publisher = self.create_publisher(Clock, "/clock", 10)
        self.clock = Clock()
        log_file = open("ident_log.csv", "w")
        self.log_file = csv.writer(log_file)
        self.declare_parameter('use_gui', True)
        self.declare_parameter('start_paused', False)
        self.declare_parameter('fixed', False)
        self.declare_parameter('pos', [0.0, 0.0, 0.4])
        self.declare_parameter('rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('robot_version', rclpy.Parameter.Type.STRING)
        self.join_state_pub = self.create_publisher(
            JointState, "/joint_states", 10)
        robot_version = self.get_parameter(
            'robot_version').get_parameter_value().string_value
        if hardware.lower() != "robot":
            use_gui = self.get_parameter(
                'use_gui').get_parameter_value().bool_value
            start_paused = self.get_parameter(
                'start_paused').get_parameter_value().bool_value
            fixed = self.get_parameter(
                'fixed').get_parameter_value().bool_value
            pos = self.get_parameter(
                'pos').get_parameter_value().double_array_value
            rpy = self.get_parameter(
                'rpy').get_parameter_value().double_array_value
            if hardware.lower() == 'mujoco':
                self.connector = MujocoConnector(robot_version, self.get_logger(),
                                                 use_gui=use_gui, start_paused=start_paused, fixed=fixed, pos=pos, rpy=rpy)
            elif hardware.lower() == 'pybullet':
                self.connector = PybulletConnector(
                    robot_version, self.get_logger(), fixed=fixed, pos=pos, rpy=rpy)
        else:
            self.connector = RobotConnector(robot_version,  self.get_logger())

    def log_data(self, t, torques, position, velocity):
        row = [0.0] * (2 + 3 * self.connector.controller.joints_num)
        if self.connector.controller.machine.is_state('move_knee', self.connector.controller):
            row[0] = 1.0
        elif self.connector.controller.machine.is_state('move_hip', self.connector.controller):
            row[0] = 2.0
        elif self.connector.controller.machine.is_state('rotate_hip', self.connector.controller):
            row[0] = 3.0
        else:
            return
        row[1] = t
        start_index = 2
        end_index = self.connector.controller.joints_num + start_index
        row[start_index:end_index] = torques.tolist()
        start_index = end_index
        end_index += self.connector.controller.joints_num
        row[start_index:end_index] = position.tolist()
        start_index = end_index
        end_index += self.connector.controller.joints_num
        row[start_index:end_index] = velocity.tolist()
        self.log_file.writerow(row)
    
    def run(self):
        c = 0
        start = self.get_clock().now()
        joint_state = JointState()
        while self.connector.is_running():
            position, velocity = self.connector.get_data()
            if self.time_publisher:
                elapsed = self.clock.clock.sec + self.clock.clock.nanosec / 1e9
            else:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            torques = self.connector.controller.compute_control(elapsed, position, velocity)
            self.connector.set_torques(torques)
            self.log_data(elapsed, torques, position, velocity)
            if self.connector.step():
                if self.time_publisher:
                    self.clock.clock.nanosec += self.connector.nanoseconds
                    self.clock.clock.sec += self.clock.clock.nanosec // 1000000000
                    self.clock.clock.nanosec = self.clock.clock.nanosec % 1000000000
                    self.time_publisher.publish(self.clock)
                c += 1
                if (c % 50 == 0):
                    if self.time_publisher:
                        joint_state.header.stamp = self.clock.clock
                    else:
                        joint_state.header.stamp = self.get_clock().now().to_msg()
                    joint_state.position = position.tolist()
                    joint_state.velocity = velocity.tolist()
                    joint_state.name = self.connector.joint_names
                    self.join_state_pub.publish(joint_state)


def main(args=None):
    rclpy.init(args=args)
    node = ConnectorNode()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
