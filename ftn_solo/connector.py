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
from geometry_msgs.msg import Twist
import time
import math
import yaml
from robot_properties_solo import Resources
from ftn_solo.tasks.robot_squat import RobotMove
from ftn_solo.tasks.task_joint_spline import TaskJointSpline
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import threading
import xacro    
import os


class Connector:
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        self.resources = Resources(robot_version)
        with open(self.resources.config_path, "r") as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as exc:
                raise exc
        self.logger = logger


class RobotConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)
        import libodri_control_interface_pywrap as oci

        self.robot = oci.robot_from_yaml_file(self.resources.config_path)
        self.joint_names = self.config["robot"]["joint_modules"]["joint_names"]
        self.robot.initialize(
            np.array([0] * self.robot.joints.number_motors, dtype=np.float64)
        )
        self.running = True
        self.dt = 0.001
        self.nanoseconds = self.dt * 1e9
        niceness = os.nice(0)
        niceness = os.nice(-20-niceness)

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
        return {"attitude": [q[3], q[0], q[1], q[2]]}


class SimulationConnector(Connector):
    def __init__(self, robot_version, logger, *args, **kwargs) -> None:
        super().__init__(robot_version, logger, *args, **kwargs)

        self.simulate_encoders = self.config.get("simulation") and self.config[
            "simulation"
        ].get("simulate_encoders", False)
        if self.simulate_encoders:
            self.resolution = (
                2
                * math.pi
                / self.config["robot"]["joint_modules"]["counts_per_revolution"]
                / self.config["robot"]["joint_modules"]["gear_ratios"]
            )
            self.old_q = None

    def process_coordinates(self, q, qdot):
        if not self.simulate_encoders:
            return q, qdot
        else:
            q = np.round(q / self.resolution) * self.resolution
            if self.old_q is None:
                self.old_q = q
                return q, 0 * qdot
            qdot = (q - self.old_q) / self.dt
            self.old_q = q
            return q, qdot


class PybulletConnector(SimulationConnector):
    def __init__(
        self,
        robot_version,
        logger,
        fixed=False,
        pos=[0, 0, 0.4],
        rpy=[0.0, 0.0, 0.0],
        *args,
        **kwargs
    ) -> None:
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
        self.touch_sensors = ["fl", "fr", "hl", "hr"]
        self.end_effector_names = ["FR_ANKLE", "FL_ANKLE", "HR_ANKLE", "HL_ANKLE"]
        self.reading = {}
        self.running = True
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        for ji in range(pybullet.getNumJoints(self.robot_id)):
            if (
                pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8")
                in self.end_effector_names
            ):
                self.end_effector_ids.append(
                    pybullet.getJointInfo(self.robot_id, ji)[0] - 1
                )
            elif pybullet.JOINT_FIXED != pybullet.getJointInfo(self.robot_id, ji)[2]:
                self.joint_names.append(
                    pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8")
                )
                self.joint_ids.append(pybullet.getJointInfo(self.robot_id, ji)[0])

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
        bodies_in_contact = set()
        contact_forces = []

        for contact_info in contact_points:
            bodies_in_contact.add(contact_info[3])
            contact_normal = contact_info[7]
            normal_force = contact_info[9]
            lateral_friction_direction_1 = contact_info[11]
            lateral_friction_force_1 = contact_info[10]
            lateral_friction_direction_2 = contact_info[13]
            lateral_friction_force_2 = contact_info[12]
            force = np.zeros(6)
            force[:3] = (
                normal_force * np.array(contact_normal)
                + lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                + lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            )
            contact_forces.append(force)

        self.reading = {
            name: self.end_effector_ids[j] in bodies_in_contact
            for j, name in enumerate(self.touch_sensors)
        }

        return self.reading

    def imu_sensor(self):
        base_inertia_pos, base_inertia_quat = pybullet.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        imu_linacc = np.cross(
            base_angvel,
            np.cross(base_angvel, rot_base_to_world @ self.r_base_to_imu),
        )

        return (
            base_inertia_quat,
            self.rot_base_to_imu.dot(rot_base_to_world.T.dot(np.array(base_angvel))),
            self.rot_base_to_imu.dot(
                rot_base_to_world.T.dot(imu_linacc + np.array([0.0, 0.0, 9.81]))
            ),
        )

    def get_sensor_readings(self):
        q, gyro, accel = self.imu_sensor()
        return {
            "attitude": [q[3], q[0], q[1], q[2]],
            "imu": (gyro, accel),
            "touch": self.contact_sensors(),
        }

    def set_torques(self, torques):
        pybullet.setJointMotorControlArray(
            self.robot_id, self.joint_ids, pybullet.TORQUE_CONTROL, forces=torques
        )

    def step(self):
        self.env.step(True)
        return True

    def is_running(self):
        return self.running

    def num_joints(self):
        return len(self.joint_names)


class MujocoConnector(SimulationConnector):
    def __init__(
        self,
        robot_version,
        logger,
        use_gui=True,
        start_paused=False,
        fixed=False,
        pos=[0, 0, 0.4],
        rpy=[0.0, 0.0, 0.0],
    ) -> None:
        super().__init__(robot_version, logger)

        xml_string = xacro.process(
            self.resources.mjcf_path + ".xacro",
            mappings={"environment": "", "resources_dir": self.resources.resources_dir},
        )
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.model.opt.timestep = 1e-3
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = pos
        mujoco.mju_euler2Quat(self.data.qpos[3:7], rpy, "XYZ")
        self.data.qpos[7:] = np.array([0.0,0.0, -1, 0.0, 0.0, -1, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        self.data.qvel[:] = 0
        if fixed:
            self.model.body("base_link").jntnum = 0
        self.joint_names = [self.model.joint(i + 1).name for i in range(self.model.nu)]
        self.paused = True
        self.use_gui = use_gui
        self.viewer = None
        self.running = True
        self.dt = self.model.opt.timestep
        self.touch_sensors = ["fl", "fr", "hl", "hr"]
        if self.use_gui:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,
                show_right_ui=False,
                key_callback=self.key_callback,
            )

    def key_callback(self, keycode):
        if chr(keycode) == " ":
            self.paused = not self.paused
        elif keycode == 256:  # ESC
            self.running = False

    def get_data(self):
        return self.process_coordinates(self.data.qpos[7:], self.data.qvel[6:])

    def get_sensor_readings(self):
        reading = {}
        for sensor in self.touch_sensors:
            name = sensor + "_touch"
            reading[name] = self.data.sensor(name).data[0] > 0
        # qw, qx, qy, qz
        return {
            "attitude": self.data.sensor("attitude").data,
            "imu": (
                self.data.sensor("angular-velocity").data,
                self.data.sensor("linear-acceleration").data,
                self.data.sensor("magnetometer").data,
            ),
            "touch": reading,
        }

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
        time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        return True

    def num_joints(self):
        return self.model.nu


class ConnectorNode(Node):
    def __init__(self):
        super().__init__("first_node")
        self.declare_parameter("hardware", rclpy.Parameter.Type.STRING)
        hardware = self.get_parameter("hardware").get_parameter_value().string_value
        self.time_publisher = None

        self.twist_subscriber = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )

        if hardware.lower() != "robot":
            self.time_publisher = self.create_publisher(Clock, "/clock", 10)
        self.clock = Clock()
        self.declare_parameter("use_gui", True)
        self.declare_parameter("start_paused", False)
        self.declare_parameter("fixed", False)
        self.declare_parameter("pos", [0.0, 0.0, 0.4])
        self.declare_parameter("rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("robot_version", rclpy.Parameter.Type.STRING)
        self.declare_parameter("task", rclpy.Parameter.Type.STRING)
        self.declare_parameter("config", rclpy.Parameter.Type.STRING)
        self.join_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        robot_version = (
            self.get_parameter("robot_version").get_parameter_value().string_value
        )
        task = self.get_parameter("task").get_parameter_value().string_value

        if hardware.lower() != "robot":
            use_gui = self.get_parameter("use_gui").get_parameter_value().bool_value
            start_paused = (
                self.get_parameter("start_paused").get_parameter_value().bool_value
            )
            fixed = self.get_parameter("fixed").get_parameter_value().bool_value
            pos = self.get_parameter("pos").get_parameter_value().double_array_value
            rpy = self.get_parameter("rpy").get_parameter_value().double_array_value
            if hardware.lower() == "mujoco":
                self.connector = MujocoConnector(
                    robot_version,
                    self.get_logger(),
                    use_gui=use_gui,
                    start_paused=start_paused,
                    fixed=fixed,
                    pos=pos,
                    rpy=rpy,
                )
            elif hardware.lower() == "pybullet":
                self.connector = PybulletConnector(
                    robot_version, self.get_logger(), fixed=fixed, pos=pos, rpy=rpy
                )
        else:
            self.connector = RobotConnector(robot_version, self.get_logger())

        if task == "joint_spline":
            self.task = TaskJointSpline(
                self.connector.num_joints(),
                robot_version,
                self.get_parameter("config").get_parameter_value().string_value,
            )
        elif task == "robot_squat":
            self.task = RobotMove(
                self.connector.num_joints(),
                robot_version,
                self.get_parameter("config").get_parameter_value().string_value,
                self.get_logger(),
                self.connector.dt,
            )

        else:
            self.get_logger().error(
                "Unknown task selected!!! Switching to joint_spline task!"
            )
            self.task = TaskJointSpline(
                robot_version,
                "/home/ajsmilutin/solo/solo_ws/src/ftn_solo/config/controllers/eurobot_demo.yaml",
            )

        self.task.dt = self.connector.dt

        self.stop_thread = False
        self.run_thread = threading.Thread(target=self.run)
        self.run_thread.start()

    def cmd_vel_callback(self, msg):
        self.task.define_movement(msg)

    def run(self):
        c = 0
        start = self.get_clock().now()
        joint_state = JointState()
        transform = TransformStamped()
        position, velocity = self.connector.get_data()
        self.task.init_pose(position, velocity)
        while rclpy.ok() and self.connector.is_running() and not self.stop_thread:
            position, velocity = self.connector.get_data()
            sensors = self.connector.get_sensor_readings()
            if self.time_publisher:
                elapsed = self.clock.clock.sec + self.clock.clock.nanosec / 1e9
            else:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            start_time = time.time()
            torques = self.task.compute_control(elapsed, position, velocity, sensors)
            self.get_logger().info(f"Time: {time.time() - start_time}")
            self.connector.set_torques(torques)
            if self.time_publisher:
                self.clock.clock.nanosec += int(self.connector.dt * 1000000000)
                self.clock.clock.sec += self.clock.clock.nanosec // 1000000000
                self.clock.clock.nanosec = self.clock.clock.nanosec % 1000000000
                self.time_publisher.publish(self.clock)
            c += 1
            if c % 50 == 0:
                if self.time_publisher:
                    joint_state.header.stamp = self.clock.clock
                else:
                    joint_state.header.stamp = self.get_clock().now().to_msg()
                joint_state.position = position.tolist()
                joint_state.velocity = velocity.tolist()
                joint_state.name = self.connector.joint_names
                self.join_state_pub.publish(joint_state)
                if "attitude" in sensors.keys():
                    transform.header.stamp = joint_state.header.stamp
                    transform.header.frame_id = "world"
                    transform.child_frame_id = "base_link"
                    transform.transform.rotation.w = sensors["attitude"][0]
                    transform.transform.rotation.x = sensors["attitude"][1]
                    transform.transform.rotation.y = sensors["attitude"][2]
                    transform.transform.rotation.z = sensors["attitude"][3]
                    self.tf_broadcaster.sendTransform(transform)
            self.connector.step()

    def destroy_node(self):
        self.stop_thread = True  # Signal the loop thread to stop
        self.run_thread.join()  # Wait for the thread to finish
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ConnectorNode()
    try:
        rclpy.spin(node)  # Handle callbacks in the main thread
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
