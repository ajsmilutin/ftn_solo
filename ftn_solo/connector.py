#!/usr/bin/env python
import sys
import mujoco
import mujoco.viewer
import pybullet
import rclpy
import pinocchio
from numpy.random import default_rng
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from rosgraph_msgs.msg import Clock
import time
import math
import yaml
from robot_properties_solo.robot_resources import Resources
from pinocchio.utils import zero

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

    def get_data(self):
        self.robot.parse_sensor_data()
        return self.robot.joints.positions, self.robot.joints.velocities

    def set_torques(self, torques):
        self.robot.joints.set_torques(torques)

    def is_rinning(self):
        return self.running

    def step(self):
        self.robot.send_command_and_wait_end_of_cycle(0.001)
        return True

class PybulletConnector(Connector):

    def __init__(self,robot_version, logger, useFixedBase=False,pos=[0, 0, 0.4],orn=[0.0, 0.0, 0.0]) -> None:
        super().__init__(robot_version, logger)
        """Initializes the wrapper.
rom pinocchio.utils import zero
        Args:
        self.joint_names = joint_names
        self.endeff_names = endeff_names

        self.base_linvel_prev = None
        self.bacchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
            joint_names (:obj:`list` of :obj:`str`): Names of the joints.
            self.urdf_path = Solo12Config.urdf_pathendeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
            useFixedBase (bool, optional): Determines if the robot base if fixed.. Defaults to False.
        """
        self.pinocchio_robot = pinocchio.buildModelFromUrdf(self.resources.mjcf_path)
        self.data=self.pinocchio_robot.createData()
        self.urdf_path = self.resources.mjcf_path
        self.robotId = pybullet.loadURDF(
            self.urdf_path,
            pos,
            orn,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=useFixedBase,
        )
        
        
            
        self.nq = self.pinocchio_robot.nq
        self.nv = self.pinocchio_robot.nv
        
        self.robot_id = self.robot_id
        self.pinocchio_robot = self.pinocchio_robot
        self.useFixedBase = useFixedBase
        self.nb_dof = self.nv - 6

        self.base_linvel_prev = None
        self.base_angvel_prev = None
        self.base_linacc = np.zeros(3)
        self.base_angacc = np.zeros(3)

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        self.rng = default_rng()

        self.base_imu_accel_bias = np.zeros(3)
        self.base_imu_gyro_bias = np.zeros(3)
        self.base_imu_accel_thermal = np.zeros(3)
        self.base_imu_gyro_thermal = np.zeros(3)
        self.base_imu_accel_thermal_noise = 0.0001962  # m/(sec^2*sqrt(Hz))
        self.base_imu_gyro_thermal_noise = 0.0000873  # rad/(sec*sqrt(Hz))
        self.base_imu_accel_bias_noise = 0.0001  # m/(sec^3*sqrt(Hz))
        self.base_imu_gyro_bias_noise = 0.000309  # rad/(sec^2*sqrt(Hz))
        
        self.end_effector_names=[]
        self.pinocchio_endeff_ids = []
        controlled_joints = []

        for leg in ["FL", "FR", "HL", "HR"]:
            controlled_joints += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
            self.pinocchio_endeff_ids.append(
                self.pinocchio_robot.getFrameId(leg + "_ANKLE")
            )
            self.end_effector_names.append(leg + "_ANKLE")
        
        self.joint_names = []
        for Jname in self.pinocchio_robot.names:
          self.joint_names.append(Jname)

        self.bullet_joint_map = {}
        for ji in range(pybullet.getNumJoints(self.robot_id)):
            self.bullet_joint_map[
                pybullet.getJointInfo(self.robot_id, ji)[1].decode("UTF-8")
            ] = ji

        self.bullet_joint_ids = np.array(
        )
        self.pinocchio_joint_ids = np.array(
            [self.pinocchio_robot.getJointId(name) for name in self.joint_names]
        )

        self.pin2bullet_joint_only_array = []

        if not self.useFixedBase:
            for i in range(2, self.nj + 2):
                self.pin2bullet_joint_only_array.append(
                    np.where(self.pinocchio_joint_ids == i)[0][0]
                )
        else:
            for i in range(1, self.nj + 1):
                self.pin2bullet_joint_only_array.append(
                    np.where(self.pinocchio_joint_ids == i)[0][0]
                )

        # Disable the velocity control on the joints as we use torque control.
        pybullet.setJointMotorControlArray(
            self.robot_id,
            self.bullet_joint_ids,
            pybullet.VELOCITY_CONTROL,
            forces=np.zeros(self.nj),
        )

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = [self.bullet_joint_map[name] for name in self.end_effector_names]
        self.end_eff_Contacts = [i-1 for i in self.bullet_endeff_ids]
       
        self.nb_contacts = len(self.pinocchio_endeff_ids)
        self.contact_status = np.zeros(self.nb_contacts)
        self.contact_forces = np.zeros([self.nb_contacts, 6])
        self.nj = len(self.joint_names)
        self.nf = len(self.end_effector_names)


    def get_joints_info(self):
        joints_ids_bullet=self.bullet_joint_ids
        joints_ids_pinnochio=self.pinocchio_joint_ids
        bullet_JointsMap=self.bullet_joint_map
        pinochioNames=self.joint_names
        pin2bullet_joints=self.pin2bullet_joint_only_array
        endeffBullet_id=self.bullet_endeff_ids
        endeffPinno_ids=self.pinocchio_endeff_ids
        
        return joints_ids_bullet,joints_ids_pinnochio,bullet_JointsMap,pin2bullet_joints,endeffBullet_id,endeffPinno_ids,pinochioNames
        
    def get_force(self):
        """Returns the force readings as well as the set of active contacts
        Returns:
            (:obj:`list` of :obj:`int`): List of active contact frame ids.
            (:obj:`list` of np.array((6,1))) List of active contact forces.
        """

        active_contacts_frame_ids = []
        contact_forces = []

        # Get the contact model using the pybullet.getContactPoints() api.
        cp = pybullet.getContactPoints()

        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
            elif ci[4] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
            else:
                continue

            if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
            force = np.zeros(6)

            force[:3] = (
                normal_force * np.array(contact_normal)
                + lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                + lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            )

            contact_forces.append(force)

        return active_contacts_frame_ids[::-1], contact_forces[::-1]

    def end_effector_forces(self):
        """Returns the forces and status for all end effectors

        Returns:
            (:obj:`list` of :obj:`int`): list of contact status for each end effector.
            (:obj:`list` of np.array(6)): List of force wrench at each end effector
        """
        contact_status = np.zeros(len(self.pinocchio_endeff_ids))
        contact_forces = np.zeros([len(self.pinocchio_endeff_ids), 6])
        # Get the contact model using the pybullet.getContactPoints() api.
        cp = pybullet.getContactPoints(self.robot_id)
       
        
        for ci in cp:
            p_ct = np.array(ci[6])
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]
            # Find id
            if ci[3] in self.end_eff_Contacts:
                i = np.where(np.array(self.end_eff_Contacts) == ci[3])[0][0]
                
            else:
                continue 
            # Contact active
            contact_status[i] = 1
            contact_forces[i, :3] += (
                normal_force * np.array(contact_normal)
                - lateral_friction_force_1 * np.array(lateral_friction_direction_1)
                - lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            )
            # there are instances when status is True but force is zero, to fix this,
            # we need the below if statement
            if np.linalg.norm(contact_forces[i, :3]) < 1.0e-12:
                contact_status[i] = 0
                contact_forces[i, :3].fill(0.0)
        return contact_status, contact_forces

    def get_base_velocity_world(self):
        """Returns the velocity of the base in the world frame.

        Returns:
            np.array((6,1)) with the translation and angular velocity
        """
        vel, orn = pybullet.getBaseVelocity(self.robot_id)
        return np.array(vel + orn).reshape(6, 1)

    def get_base_acceleration_world(self):
        """Returns the numerically-computed acceleration of the base in the world frame.

        Returns:
            np.array((6,1)) vector of linear and angular acceleration
        """
        return np.concatenate((self.base_linacc, self.base_angacc))

    def get_base_imu_angvel(self):
        """Returns simulated base IMU gyroscope angular velocity.

        Returns:
            np.array((3,1)) IMU gyroscope angular velocity (base frame)
        """
        base_inertia_pos, base_inertia_quat = pybullet.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        return (
            self.rot_base_to_imu.dot(rot_base_to_world.T.dot(np.array(base_angvel)))
            + self.base_imu_gyro_bias
            + self.base_imu_gyro_thermal
        )

    def get_base_imu_linacc(self):
        """Returns simulated base IMU accelerometer acceleration.

        Returns:
            np.array((3,1)) IMU accelerometer acceleration (base frame, gravity offset)
        """
        base_inertia_pos, base_inertia_quat = pybullet.getBasePositionAndOrientation(
            self.robot_id
        )
        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_inertia_quat)
        ).reshape((3, 3))
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        # Transform the base acceleration to the IMU position, in world frame
        imu_linacc = (
            self.base_linacc
            + np.cross(self.base_angacc, rot_base_to_world @ self.r_base_to_imu)
            + np.cross(
                base_angvel,
                np.cross(base_angvel, rot_base_to_world @ self.r_base_to_imu),
            )
        )

        return (
            self.rot_base_to_imu.dot(
                rot_base_to_world.T.dot(imu_linacc + np.array([0.0, 0.0, 9.81]))
            )
            + self.base_imu_accel_bias
            + self.base_imu_accel_thermal
        )

    def get_data(self):
        """Returns a pinocchio-like representation of the q, dq matrices. Note that the base velocities are expressed in the base frame.

        Returns:
            ndarray: Generalized positions.
            ndarray: Generalized velocities.
        """

        q = zero(self.nq)
        dq = zero(self.nv)

        if not self.useFixedBase:
            (
                base_inertia_pos,
                base_inertia_quat,
            ) = pybullet.getBasePositionAndOrientation(self.robot_id)
            # Get transform between inertial frame and link frame in base
            base_stat = pybullet.getDynamicsInfo(self.robot_id, -1)
            base_inertia_link_pos, base_inertia_link_quat = pybullet.invertTransform(
                base_stat[3], base_stat[4]
            )
            pos, orn = pybullet.multiplyTransforms(
                base_inertia_pos,
                base_inertia_quat,
                base_inertia_link_pos,
                base_inertia_link_quat,
            )
            q[:3] = pos
            q[3:7] = orn

            vel, orn = pybullet.getBaseVelocity(self.robot_id)
            dq[:3] = vel
            dq[3:6] = orn

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot = np.array(pybullet.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            dq[0:3] = rot.T.dot(dq[0:3])
            dq[3:6] = rot.T.dot(dq[3:6])

        # Query the joint readings.
        joint_states = pybullet.getJointStates(self.robot_id, self.bullet_joint_ids)

        if not self.useFixedBase:
            for i in range(self.nj):
                q[5 + self.pinocchio_joint_ids[i]] = joint_states[i][0]
                dq[4 + self.pinocchio_joint_ids[i]] = joint_states[i][1]
        else:
            for i in range(self.nj):
                q[self.pinocchio_joint_ids[i] - 1] = joint_states[i][0]
                dq[self.pinocchio_joint_ids[i] - 1] = joint_states[i][1]

        return q, dq

    def get_imu_frame_position_velocity(self):
        """Returns the position and velocity of IMU frame. Note that the velocity is expressed in the IMU frame.

        Returns:
            np.array((3,1)): IMU frame position expressed in world.
            np.array((3,1)): IMU frame velocity expressed in IMU frame.
        """
        base_pose, base_quat = pybullet.getBasePositionAndOrientation(self.robot_id)
        base_linvel, base_angvel = pybullet.getBaseVelocity(self.robot_id)

        rot_base_to_world = np.array(
            pybullet.getMatrixFromQuaternion(base_quat)
        ).reshape((3, 3))
        rot_imu_to_world = rot_base_to_world.dot(self.rot_base_to_imu.T)

        imu_position = base_pose + rot_base_to_world.dot(self.r_base_to_imu)
        imu_velocity = rot_imu_to_world.T.dot(
            base_linvel
            + np.cross(base_angvel, rot_base_to_world.dot(self.r_base_to_imu))
        )
        return imu_position, imu_velocity

 
    def reset_state(self, q, dq):
        """Reset the robot to the desired states.

        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

        if not self.useFixedBase:
            # Get transform between inertial frame and link frame in base
            base_stat = pybullet.getDynamicsInfo(self.robot_id, -1)
            base_pos, base_quat = pybullet.multiplyTransforms(
                vec2list(q[:3]), vec2list(q[3:7]), base_stat[3], base_stat[4]
            )
            pybullet.resetBasePositionAndOrientation(self.robot_id, base_pos, base_quat)

            # Pybullet assumes the base velocity to be aligned with the world frame.
            rot = np.array(pybullet.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            pybullet.resetBaseVelocity(
                self.robot_id, vec2list(rot.dot(dq[:3])), vec2list(rot.dot(dq[3:6]))
            )

            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                pybullet.resetJointState(
                    self.robot_id,
                    bullet_joint_id,
                    q[5 + self.pinocchio_joint_ids[i]],
                    dq[4 + self.pinocchio_joint_ids[i]],
                )
        else:
            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                pybullet.resetJointState(
                    self.robot_id,
                    bullet_joint_id,
                    q[self.pinocchio_joint_ids[i] - 1],
                    dq[self.pinocchio_joint_ids[i] - 1],
                )

    def set_torques(self, tau):
        """Apply the desired torques to the joints.

        Args:
            tau (ndarray): Torque to be applied.
        """
        # TODO: Apply the torques on the base towards the simulator as well.
        if not self.useFixedBase:
            assert tau.shape[0] == self.nv - 6
        else:
            assert tau.shape[0] == self.nv

        zeroGains = tau.shape[0] * (0.0,)

        pybullet.setJointMotorControlArray(
            self.robot_id,
            self.bullet_joint_ids,
            pybullet.TORQUE_CONTROL,
            forces=tau[self.pin2bullet_joint_only_array],
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )



    def print_physics_params(self):
        """Print physics engine parameters."""
        # Query all the joints.
        num_joints = pybullet.getNumJoints(self.robot_id)

        for ji in range(num_joints):
            (
                mass,
                lateral_friction,
                local_inertia_diag,
                local_inertia_pos,
                local_inertia_ori,
                resitution,
                rolling_friction,
                spinning_friction,
                contact_damping,
                contact_stiffness,
            ) = pybullet.getDynamicsInfo(bodyUniqueId=self.robot_id, linkIndex=ji)
            # for el in dynamics_info:
            #     print(el)
            print("link ", ji)
            print("    - mass : ", mass)
            print("    - lateral_friction : ", lateral_friction)
            print("    - local_inertia_diag : ", local_inertia_diag)
            print("    - local_inertia_pos : ", local_inertia_pos)
            print("    - local_inertia_ori : ", local_inertia_ori)
            print("    - resitution : ", resitution)
            print("    - rolling_friction : ", rolling_friction)
            print("    - spinning_friction : ", spinning_friction)
            print("    - contact_damping : ", contact_damping)
            print("    - contact_stiffness : ", contact_stiffness)
    
    def step(self):
        """Step the simulation forward."""
        pybullet.stepSimulation()
        return True
    
    def is_rinning(self):
        return self.running

        
class MujocoConnector(Connector):
    def __init__(self, robot_version, logger, use_gui=True, start_paused=False, fixed=False, pos=[0, 0, 0.4], rpy=[0.0, 0.0, 0.0]) -> None:
        super().__init__(robot_version, logger)
        self.model = mujoco.MjModel.from_xml_path(self.resources.mjcf_path)
        self.model.opt.timestep = 1e-3
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:3] = pos
        logger.error(str(rpy))
        self.data.qpos[3:7] = RPY2Quat(rpy)
        logger.error(str(self.data.qpos))
        self.data.qpos[7:] = 0
        if fixed:
            self.model.body("base_link").jntnum = 0
        self.joint_names = [self.model.joint(
            i+1).name for i in range(self.model.nu)]
        self.paused = start_paused
        self.use_gui = use_gui
        self.viewer = None
        self.running = True
        self.ns = int(self.model.opt.timestep*1e9)

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

    def is_rinning(self):
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
        self.declare_parameter('sim',rclpy.Parameter.Type.STRING)
        sim = self.get_parameter('sim').get_parameter_value().string_value
        self.time_publisher = None
        if sim:
            self.time_publisher = self.create_publisher(Clock, "/clock", 10)
        self.clock = Clock()
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
        if sim:
            use_gui = self.get_parameter(
                'use_gui').get_parameter_value().bool_value
            start_paused = self.get_parameter(
                'start_paused').get_parameter_value().bool_value
            fixed = self.get_parameter(
                'fixed').get_parameter_value().bool_value
            pos = self.get_parameter('pos').get_parameter_value().double_array_value
            rpy = self.get_parameter('rpy').get_parameter_value().double_array_value
            if sim=='mujoco':
                self.connector = MujocoConnector(robot_version, self.get_logger(),
                                             use_gui=use_gui, start_paused=start_paused, fixed=fixed, pos=pos, rpy=rpy)
            elif sim=='pybullet':
                self.connector = PybulletConnector(robot_version, self.get_logger(), fixed=fixed, pos=pos, rpy=rpy)
        else:
            self.connector = RobotConnector(robot_version,  self.get_logger())

    def run(self):
        c = 0
        des_pos = np.array(
            [0.0, 0, -1.57, 0, 0, -1.57, 0.3, 0.9, -1.57, -0.3, 0.9, -1.57])
        start = self.get_clock().now()
        joint_state = JointState()
        while self.connector.is_rinning():
            position, velocity = self.connector.get_data()
            if self.time_publisher:
                elapsed = self.clock.clock.sec + self.clock.clock.nanosec / 1e9
            else:
                elapsed = (self.get_clock().now() - start).nanoseconds / 1e9

            torques = 5 * (des_pos*0.5*(1-math.cos(5*elapsed)) - position) + \
                0.00725 * (des_pos*0.5*math.sin(5*elapsed) - velocity)
            self.connector.set_torques(torques)
            if self.connector.step():
                if self.time_publisher:
                    self.clock.clock.nanosec += self.connector.ns
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
