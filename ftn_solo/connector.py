#!/usr/bin/env python
import sys
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
from robot_properties_solo import Resources
from ftn_solo.tasks import *
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ftn_solo.utils.conversions import ToVector
from ftn_solo_control import SensorData
import xacro
import os
from ament_index_python.packages import get_package_share_directory

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.alarm(0)



class Connector():
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        self.resources = Resources(robot_version)
        with open(self.resources.config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc
        self.logger = logger

    def is_paused(self):
        return False


class RobotConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)
        import libodri_control_interface_pywrap as oci
        self.robot = oci.robot_from_yaml_file(self.resources.config_path)
        self.joint_names = self.config["robot"]["joint_modules"]["joint_names"]
        self.robot.initialize(
            np.array([0]*self.robot.joints.number_motors, dtype=np.float64))
        self.running = True
        self.dt = 0.0010001
        self.nanoseconds = self.dt*1e9

    def get_data(self):
        self.robot.parse_sensor_data()
        return self.robot.joints.positions, self.robot.joints.velocities

    def set_torques(self, torques):
        self.robot.joints.set_torques(torques)

    def is_running(self):
        if self.robot.has_error:
            self.logger.error("Error appeared")
        if self.robot.is_timeout:
            self.logger.error("Timeout happened with real robot")
        return not (self.robot.has_error)

    def step(self):
        self.robot.send_command_and_wait_end_of_cycle(self.dt)
        return True

    def num_joints(self):
        return self.robot.joints.number_motors

    def get_sensor_readings(self):
        q = self.robot.imu.attitude_quaternion
        data = SensorData()
        data.imu_data.attitude = np.array([q[3], q[0], q[1], q[2]])
        data.imu_data.angular_velocity =self.robot.imu.gyroscope
        return data


class SimulationConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)

        self.simulate_encoders = self.config.get(
            "simulation") and self.config["simulation"].get("simulate_encoders", False)
        if self.simulate_encoders:
            self.resolution = 2*math.pi / \
                self.config["robot"]["joint_modules"]["counts_per_revolution"] / \
                self.config["robot"]["joint_modules"]["gear_ratios"]
            self.old_q = None

    def process_coordinates(self, q, qdot):
        if not self.simulate_encoders:
            return q, qdot
        else:
            q = np.round(q / self.resolution) * self.resolution
            if self.old_q is None:
                self.old_q = q
                return q, 0*qdot
            qdot = (q - self.old_q)/self.dt
            self.old_q = q
            return q, qdot


class PybulletConnector(SimulationConnector):
    def __init__(self, robot_version, logger, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0], *args, **kwargs) -> None:
        super().__init__(robot_version, logger)

        self.env = BulletEnvWithGround(robot_version)
        orn = pybullet.getQuaternionFromEuler(rpy)
        self.dt = self.env.dt
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
        self.touch_sensors = ['fl', 'fr', 'hl', 'hr']
        self.end_effector_names = ['FR_ANKLE',
                                   'FL_ANKLE', 'HR_ANKLE', 'HL_ANKLE']
        self.reading = {}
        self.running = True
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        for ji in range(pybullet.getNumJoints(self.robot_id)):
            if pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8") in self.end_effector_names:
                self.end_effector_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0]-1)
            elif pybullet.JOINT_FIXED != pybullet.getJointInfo(self.robot_id, ji)[2]:
                self.joint_names.append(
                    pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8"))
                self.joint_ids.append(
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

        return self.process_coordinates(q, dq)

    def contact_sensors(self):
        contact_points = pybullet.getContactPoints(self.robot_id)
        bodies_in_contact = list()

        for contact_info in contact_points:
            bodies_in_contact.add(contact_info[3])

        self.reading = [self.end_effector_ids[j] in bodies_in_contact for j, name in enumerate(
            self.touch_sensors)]

        return self.reading

    def imu_sensor(self):
        base_inertia_pos, base_inertia_quat = pybullet.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        imu_linacc = (
            np.cross(
                base_angvel,
                np.cross(base_angvel, rot_base_to_world @ self.r_base_to_imu),
            )
        )

        return (base_inertia_quat,
                self.rot_base_to_imu.dot(
                    rot_base_to_world.T.dot(np.array(base_angvel))),
                self.rot_base_to_imu.dot(
                    rot_base_to_world.T.dot(imu_linacc + np.array([0.0, 0.0, 9.81]))))

    def get_sensor_readings(self):
        readings = SensorData()
        q, gyro, accel = self.imu_sensor()

        readings.touch[i] = np.array(self.contact_sensors(), dtype=np.bool)
        readings.imu_data.attitude = np.array([q[3], q[0], q[1], q[2]])
        readings.imu_data.angular_velocity = gyro
        readings.imu_data.linear_acceleration = accel

        return readings

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

    def num_joints(self):
        return len(self.joint_names)


class MujocoConnector(SimulationConnector):
    def __init__(self, robot_version, logger, use_gui=True, start_paused=False, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0], environment="", environments_package="") -> None:
        super().__init__(robot_version, logger)
        with open(self.resources.mjcf_path, 'r') as file:
            xml_string = file.read()

        environment_path = ""
        if not environment == "":
            if not environments_package == "":
                environment_path = os.path.join(
                    get_package_share_directory(environments_package), environment)
            else:
                environment_path = environment
        xml_string = xacro.process(self.resources.mjcf_path + ".xacro", mappings={
                                   "environment": environment_path, "resources_dir": self.resources.resources_dir})
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.model.opt.timestep = 0.0010001

        if fixed:
            self.model.equality("fixed").active0 = True
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = pos
        mujoco.mju_euler2Quat(self.data.qpos[3:7], rpy, "XYZ")
        self.data.qpos[7:] = 0
        # UNITREEE
        # self.data.qpos[8::3] = 0.75
        # self.data.qpos[9::3] = -1.5
    

        self.data.qvel[:] = 0
        self.joint_names = [self.model.joint(
            i+1).name for i in range(self.model.nu)]
        self.paused = start_paused
        self.use_gui = use_gui
        self.viewer = None
        self.running = True
        self.dt = self.model.opt.timestep
        self.touch_sensors = ["fl", "fr", "hl", "hr"]
        if self.use_gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=self.key_callback)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=True

    def key_callback(self, keycode):
        if chr(keycode) == ' ':
            self.paused = not self.paused
        elif keycode == 256:  # ESC
            self.running = False

    def get_data(self):
        return self.process_coordinates(self.data.qpos[7:], self.data.qvel[6:])

    def get_sensor_readings(self):
        readings = SensorData()
        for i, sensor in enumerate(self.touch_sensors):
            name = sensor + "_touch"
            readings.touch[i] = self.data.sensor(name).data[0] > 0
        readings.imu_data.attitude = self.data.sensor("attitude").data
        readings.imu_data.angular_velocity = self.data.sensor(
            "angular-velocity").data
        readings.imu_data.linear_acceleration = self.data.sensor(
            "linear-acceleration").data
        readings.imu_data.magnetometer = self.data.sensor("magnetometer").data
        # qw, qx, qy, qz
        return readings

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

    def num_joints(self):
        return self.model.nu

    def is_paused(self):
        return self.paused


class ConnectorNode(Node):
    def __init__(self):
        super().__init__("first_node")
        self.declare_parameter('hardware', rclpy.Parameter.Type.STRING)
        hardware = self.get_parameter(
            'hardware').get_parameter_value().string_value
        self.time_publisher = None
        if hardware.lower() != "robot":
            self.time_publisher = self.create_publisher(Clock, "/clock", 10)
        self.clock = Clock()
        self.declare_parameter('use_gui', True)
        self.declare_parameter('start_paused', False)
        self.declare_parameter('fixed', False)
        self.declare_parameter('pos', [0.0, 0.0, 0.4])
        self.declare_parameter('rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('robot_version', rclpy.Parameter.Type.STRING)
        self.declare_parameter('task', rclpy.Parameter.Type.STRING)
        self.declare_parameter('config', rclpy.Parameter.Type.STRING)
        self.join_state_pub = self.create_publisher(
            JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        robot_version = self.get_parameter(
            'robot_version').get_parameter_value().string_value
        task = self.get_parameter(
            'task').get_parameter_value().string_value

        self.fixed = False
        yaml_config = self.get_parameter(
            'config').get_parameter_value().string_value
        with open(yaml_config) as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc

        if hardware.lower() != "robot":
            use_gui = self.get_parameter(
                'use_gui').get_parameter_value().bool_value
            start_paused = self.get_parameter(
                'start_paused').get_parameter_value().bool_value
            self.fixed = self.get_parameter(
                'fixed').get_parameter_value().bool_value
            pos = self.get_parameter(
                'pos').get_parameter_value().double_array_value
            rpy = self.get_parameter(
                'rpy').get_parameter_value().double_array_value
            # Can go lower if we set niceness
            self.allowed_time = 0.1
            if hardware.lower() == 'mujoco':
                self.connector = MujocoConnector(robot_version, self.get_logger(),
                                                 pos=pos, rpy=rpy, **self.config["mujoco"])
            elif hardware.lower() == 'pybullet':
                self.connector = PybulletConnector(
                    robot_version, self.get_logger(), fixed=self.fixed, pos=pos, rpy=rpy)
        else:
            niceness = os.nice(0)
            niceness = os.nice(-20-niceness)
            self.get_logger().info("Setting niceness to {}".format(niceness))
            self.allowed_time = 1.0
            self.connector = RobotConnector(robot_version,  self.get_logger())

        self.get_logger().info("Allowed time to run is {}".format(self.allowed_time))
        if task == 'joint_spline':
            self.task = TaskJointSpline(self.connector.num_joints(),
                                        robot_version, self.config)
        elif task == 'move_base':
            self.task = TaskMoveBase(self.connector.num_joints(),
                                     robot_version, self.config)
        elif task == 'draw_shapes':
            self.task = TaskDrawShapes(self.connector.num_joints(),
                                       robot_version, self.config)
        elif task == 'friction_identification':
            self.task = TaskFrictionIdentification(self.connector.num_joints(
            ),                                       robot_version, self.config)
        else:
            self.get_logger().error(
                'Unknown task selected!!! Switching to joint_spline task!')
            self.task = TaskJointSpline(
                robot_version, "/home/ajsmilutin/solo/solo_ws/src/ftn_solo/config/controllers/eurobot_demo.yaml")
        self.task.dt = self.connector.dt

    def run(self):
        c = 0
        start = self.get_clock().now()
        joint_state = JointState()
        transform = TransformStamped()
        position, velocity = self.connector.get_data()
        self.task.init_pose(position, velocity)
        num_rows = 50000
        # all_data=np.ndarray((num_rows, 12+12+12+6+1), dtype=np.float64)
        # print(all_data)
        while self.connector.is_running():
            start_time = time.time()
            # if c == num_rows:
                # np.savetxt("/home/ajsmilutin/solo_ws/data_sim.csv", all_data, delimiter=",")
            if self.connector.is_paused():
                continue
            position, velocity = self.connector.get_data()
            # all_data[c, 0] = time.time()-start_time
            # all_data[c, 7:19] = position
            # all_data[c, 19:31] = velocity
            sensors = self.connector.get_sensor_readings()
            # all_data[c, 1] = time.time()-start_time
            if self.time_publisher:
                elapsed = self.clock.clock.sec + self.clock.clock.nanosec / 1e9
            else:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            try:
                with time_limit(10000):                    
                    torques = self.task.compute_control(
                        elapsed, position, velocity, sensors)
                    # all_data[c, 2] = time.time()-start_time                    
                    if self.time_publisher:
                        self.clock.clock.nanosec += int(self.connector.dt * 1000000000)
                        self.clock.clock.sec += self.clock.clock.nanosec // 1000000000
                        self.clock.clock.nanosec = self.clock.clock.nanosec % 1000000000
                        # self.time_publisher.publish(self.clock)
                    c += 1
                    if (c % 20 == 0):
                        if self.time_publisher:
                            stamp = self.clock.clock
                        else:
                            stamp = self.get_clock().now().to_msg()
                            
                        if hasattr(self.task, "estimator"):
                            if self.task.estimator and self.task.estimator.initialized():
                                self.task.estimator.publish_state(stamp.sec, stamp.nanosec)
                                # all_data[c-1, 31:43] = self.task.estimator.estimated_qv[-12:]
                        else:
                            joint_state.header.stamp = stamp
                            joint_state.position = position.tolist()
                            joint_state.velocity = velocity.tolist()
                            joint_state.name = self.connector.joint_names
                            # self.join_state_pub.publish(joint_state)
                            transform.header.stamp = joint_state.header.stamp
                            transform.header.frame_id = "world"
                            transform.child_frame_id = "base_link"
                            if self.fixed:
                                transform.transform.translation.z = 0.7
                            transform.transform.rotation.w = sensors.imu_data.attitude[0]
                            transform.transform.rotation.x = sensors.imu_data.attitude[1]
                            transform.transform.rotation.y = sensors.imu_data.attitude[2]
                            transform.transform.rotation.z = sensors.imu_data.attitude[3]
                            # self.tf_broadcaster.sendTransform(transform)
                    # all_data[c-1, 3] = time.time()-start_time                            
            except TimeoutException as e:
                self.get_logger().error("====== TIMED OUT! ======")
                exit()
            self.connector.set_torques(torques)
            # all_data[c-1, 4] = time.time()-start_time
            self.connector.step()
            # all_data[c-1, 5] = time.time()-start_time
            # all_data[c-1, 6] = elapsed
            print("Elapsed {}".format(elapsed))


def main(args=None):
    rclpy.init(args=args)
    node = ConnectorNode()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
