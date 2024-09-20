from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import PDWithFrictionCompensation
from robot_properties_solo import Solo12Robot
from geometry_msgs.msg import Point, Vector3
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint, ToQuaternion, ToVector
from copy import deepcopy
from scipy.special import erf
from ftn_solo.utils.trajectories import get_trajectory_marker, SplineData
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import time as tm
import math
import proxsuite
from ftn_solo_control import (
    FrictionCone,
    FixedPointsEstimator,
    FrictionConeMap,
    PieceWiseLinearPosition,
    PieceWiseLinearRotation,
    SplineTrajectory,
    publish_cone_marker,
    get_touching_pose,
    get_touching_placement,
    get_end_of_motion,
    get_end_of_motion_prioritized,
    EEFPositionMotion,
    EEFRotationMotion,
    COMMotion,
    WholeBodyController,
    MotionsVector,
    get_projected_wcm,
    get_projected_wcm_with_torque,
    intersect
)


class MotionData:
    def __init__(self, config):
        self.eef_index = config["eef"]
        self.positions = []
        if "positions" in config:
            for position in config["positions"]:
                self.positions.append(np.array(position))
            self.times = config["times"]
        elif "position" in config:
            self.positions = [np.array(config["position"])]
        else:
            self.positions = [np.arry([0.0, 0.0, 0.0])]

        quaternion = config["orientation"] if "orientation" in config else [
            0.0, 0.0, 0.0, 1.0]
        self.rotation = pin.XYZQUATToSE3(
            self.positions[0].tolist() + quaternion).rotation


class Phase:
    def __init__(self, order, config):
        self.order = order
        self.duration = config["duration"]
        self.torso_height = None if not "torso_height" in config else config["torso_height"]
        self.motions = []
        for motion_cfg in config["motions"]:
            self.motions.append(MotionData(motion_cfg))


def parse_sequence(config):
    sequence = []
    for i, phase_config in enumerate(config):
        sequence.append(Phase(i, phase_config))
    return sequence


class TaskMoveBase(TaskBase):
    states = ["start", "move_base", "move_eef"]

    def __init__(self, num_joints, robot_type, config_yaml) -> None:
        self.step = 0
        super().__init__(num_joints, robot_type, config_yaml)
        if robot_type == "solo12":
            self.robot = Solo12Robot()
        else:
            raise ("Only solo12 supported")
        self.joint_controller = PDWithFrictionCompensation(
            self.robot.pin_robot, self.config["joint_controller"]
        )
        self.parse_poses(self.config["poses"])
        self.on_start = SplineData(
            self.config["on_start"], self.num_joints, self.poses)
        self.machine = Machine(
            model=self, states=TaskMoveBase.states, initial="start")

        self.machine.add_transition(
            "tick", "start", "move_base", conditions="following_spline"
        )
        self.machine.add_transition(
            "tick", "move_base", "move_eef", conditions="moving_base")

        self.machine.add_transition(
            "tick", "move_base", "move_base", conditions="always_false"
        )
        self.machine.add_transition(
            "tick", "move_eef", "move_base", conditions="moving_eef"
        )

        self.machine.on_enter_move_base(self.compute_base_trajectory)
        self.machine.on_enter_move_eef(self.update_eef_trajectory)

        self.node = Node("node")
        self.estimator = None
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 10)
        self.pose_publisher = self.node.create_publisher(
            PoseArray, "origin_pose", 10)
        self.join_state_pub = self.node.create_publisher(
            JointState, "/ik/joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 8
        self.friction_coefficient = 0.9
        self.torso_height = 0.2
        self.friction_cones = dict()
        self.sequence = parse_sequence(self.config["crawl"])
        self.phase = -1
        self.base_motions = MotionsVector()
        self.eef_motions = MotionsVector()
        self.motions = MotionsVector()
        self.end_times = {}
        self.ik_data = pin.Data(self.robot.pin_robot.model)
        self.max_torque = 2.0

    def parse_poses(self, poses):
        self.poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(
                poses[pose_name], dtype=np.float64)

    def init_pose(self, q, qv):
        self.compute_trajectory(0, q, self.on_start)

    def compute_trajectory(self, tstart, q, sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = CubicSpline(
            np.hstack(([tstart], tstart + sequence.durations)),
            np.vstack((q, sequence.poses)),
            bc_type="clamped",
        )
        self.last_pose = sequence.poses[-1, :]

    def update_phase(self, t, q, qv, sensors):
        self.phase = self.phase + 1
        if self.phase == 0:
            self.estimator = FixedPointsEstimator(
                0.001,
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                self.robot.end_eff_ids,
            )
            self.estimator.init(q, qv, sensors)
            self.estimator.estimate(t, q, qv, sensors)
            self.robot.forward_robot(
                self.estimator.estimated_q, self.estimator.estimated_qv
            )
        else:
            phase = self.phase - 1
            for motion in self.sequence[phase].motions:
                self.estimator.set_fixed(
                    motion.eef_index, motion.rotation)
        self.friction_cones = self.estimator.get_friction_cones(
            self.friction_coefficient, self.num_faces)

    def always_false(self, *args, **kwargs):
        return False

    def publish_wcm_markers(self, wcm_list, next_com, prefix="wcm_"):
        for cone in self.friction_cones:
            publish_cone_marker(cone.data())
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.5)
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = 0.005
        marker.id = 205
        marker.ns = "wcm"
        marker.pose.position = Point(x=0.0, y=0.0, z=0.0)
        marker.pose.orientation.x = marker.pose.orientation.y = (
            marker.pose.orientation.z
        ) = 0.0
        marker.pose.orientation.w = 1.0
        for i, wcm_2d in enumerate(wcm_list):
            marker2 = deepcopy(marker)
            marker2.id = 205+i
            marker2.points.clear()
            marker2.color.g = marker2.color.g/(2**i)
            marker2.ns = prefix+str(i)
            for wcm_point in wcm_2d.points:
                point = Point()
                point.z = 0.0
                point.x = wcm_point[0]
                point.y = wcm_point[1]
                marker2.points.append(point)
            marker2.points.append(marker2.points[0])
            markers.markers.append(marker2)

        marker3 = deepcopy(marker)
        marker3.ns = prefix+"com"
        marker3.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        marker3.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker3.id = 207
        marker3.type = Marker.SPHERE
        marker3.pose.position.x = next_com[0]
        marker3.pose.position.y = next_com[1]
        markers.markers.append(marker3)
        self.publisher.publish(markers)

    def compute_com_xy(self, publish_markers=True):
        wcm_2d = get_projected_wcm(self.friction_cones)

        self.next_friction_cones = FrictionConeMap()
        for f in self.friction_cones:
            self.next_friction_cones[f.key()] = f.data()
        if self.phase < len(self.sequence):
            for motion in self.sequence[self.phase].motions:
                del self.next_friction_cones[motion.eef_index]
        next_wcm_2d = get_projected_wcm(self.next_friction_cones)
        return self.compute_com_pos(wcm_2d, next_wcm_2d, publish_markers=publish_markers, prefix="wcm_")

    def compute_com_xy_with_torque(self, publish_markers=True):
        wcm_2d = get_projected_wcm_with_torque(
            self.robot.pin_robot.model, self.ik_data, self.friction_cones, self.max_torque)
        next_wcm_2d = get_projected_wcm_with_torque(
            self.robot.pin_robot.model, self.ik_data, self.next_friction_cones, self.max_torque)
        return self.compute_com_pos(wcm_2d, next_wcm_2d, publish_markers=publish_markers,  prefix="wct_")

    def compute_com_pos(self,  wcm_2d, next_wcm_2d, publish_markers, prefix):
        intersection = next_wcm_2d
        C = np.copy(intersection.equations())
        d = 0.03 - C[:, 2]
        C = C[:, :2]
        H = np.eye(2)
        qp = proxsuite.proxqp.dense.QP(
            2, 0, C.shape[0], False, proxsuite.proxqp.dense.HessianType.Dense)
        # if self.eef_des is not None:
        #     desp = self.eef_des
        # else:
        desp = self.robot.pin_robot.data.com[0]
        g = - np.copy(desp[:2])

        qp.init(
            H,
            g,
            None,
            None,
            C,
            d,
            1e20 * np.ones(C.shape[0])
        )
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        x0 = np.mean(np.column_stack(intersection.points), axis=1)
        # C = np.copy(intersection.equations())
        # d = 0.02 - C[:, 2]
        # C[:, 2] = 1
        # qp = proxsuite.proxqp.dense.QP(
        #     3, 0, C.shape[0], False, proxsuite.proxqp.dense.HessianType.Zero
        # )
        # qp.init(
        #     np.zeros((3, 3)),
        #     np.array([0, 0, 1]),
        #     None,
        #     None,
        #     C,
        #     d,
        #     1e20 * np.ones(C.shape[0]),
        # )
        # qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START
        # x0 = np.zeros((3))
        # x0[:2] = np.mean(np.column_stack(intersection.points), axis=1)
        qp.solve(x0, None, None)
        pos = qp.results.x[:2]
        if publish_markers:
            self.publish_wcm_markers(
                [wcm_2d, next_wcm_2d, intersection], pos, prefix=prefix)
        return pos

    def get_new_origin(self):
        if len(self.friction_cones) == 4:
            x = np.copy(
                self.friction_cones[self.robot.end_eff_ids[0]].get_position())
            x = x + \
                self.friction_cones[self.robot.end_eff_ids[1]].get_position()
            x = x - \
                self.friction_cones[self.robot.end_eff_ids[2]].get_position()
            x = x - \
                self.friction_cones[self.robot.end_eff_ids[3]].get_position()
            y = np.copy(
                self.friction_cones[self.robot.end_eff_ids[0]].get_position())
            y = y - \
                self.friction_cones[self.robot.end_eff_ids[1]].get_position()
            y = y + \
                self.friction_cones[self.robot.end_eff_ids[2]].get_position()
            y = y - \
                self.friction_cones[self.robot.end_eff_ids[3]].get_position()
        else:
            if (
                self.robot.end_eff_ids[0] in self.friction_cones
                and self.robot.end_eff_ids[2] in self.friction_cones
            ):
                x = self.friction_cones[self.robot.end_eff_ids[0]].get_position(
                )
                x = x - \
                    self.friction_cones[self.robot.end_eff_ids[2]
                                        ].get_position()

            else:
                x = self.friction_cones[self.robot.end_eff_ids[1]].get_position(
                )
                x = x - \
                    self.friction_cones[self.robot.end_eff_ids[3]
                                        ].get_position()

            if (
                self.robot.end_eff_ids[0] in self.friction_cones
                and self.robot.end_eff_ids[1] in self.friction_cones
            ):
                y = self.friction_cones[self.robot.end_eff_ids[0]].get_position(
                )
                y = y - \
                    self.friction_cones[self.robot.end_eff_ids[1]
                                        ].get_position()
            else:
                y = self.friction_cones[self.robot.end_eff_ids[2]].get_position(
                )
                y = y - \
                    self.friction_cones[self.robot.end_eff_ids[3]
                                        ].get_position()

        origin = np.zeros(3)
        for cone in self.friction_cones:
            origin = origin + cone.data().get_position()
        origin = origin / len(self.friction_cones)
        x = x / np.linalg.norm(x)
        z = np.cross(x, y)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        rot = np.column_stack((x, y, z))
        return pin.SE3(rot, origin)

    def publish_pose(self, pose):
        poses_msg = PoseArray()
        poses_msg.header.frame_id = "world"
        pose_msg = Pose()
        pose_msg.position = ToPoint(pose.translation)
        pose_msg.orientation = ToQuaternion(
            pin.Quaternion(pose.rotation))
        poses_msg.poses.append(pose_msg)
        self.pose_publisher.publish(poses_msg)

    def compute_base_trajectory(self, t, q, qv, sensors):
        self.update_phase(t, q, qv, sensors)
        time = self.sequence[self.phase].duration
        # create base trajectory
        pos = self.compute_com_xy()
        self.origin_pose = self.get_new_origin()
        self.publish_pose(self.origin_pose)
        com_trajectory = PieceWiseLinearPosition()
        com_trajectory.add(np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        com_trajectory.add(pos, time)
        self.com_motion = COMMotion(
            np.array([True, True, False]), pin.SE3.Identity(), 200, 200)
        self.com_motion.set_trajectory(com_trajectory)

        if self.sequence[self.phase].torso_height:
            self.torso_height = self.sequence[self.phase].torso_height
        base_pose = self.robot.pin_robot.data.oMf[self.base_index]
        base_trajectory = PieceWiseLinearPosition()
        base_trajectory.add(
            np.array([self.origin_pose.actInv(base_pose).translation[2]]), 0.0)

        base_trajectory.add(
            np.array([self.torso_height]), time)

        self.base_linear_motion = EEFPositionMotion(
            self.base_index, np.array(
                [False, False, True], dtype=bool), self.origin_pose, 400, 250
        )
        self.base_linear_motion.set_trajectory(base_trajectory)
        self.base_linear_motion.set_priority(1, 1.0)

        self.base_angular_motion = EEFRotationMotion(self.base_index, 400, 200)
        self.base_angular_motion.set_priority(1, 1.0)
        rotation_trajectory = PieceWiseLinearRotation()
        rotation_trajectory.add(base_pose.rotation, 0)
        rotation_trajectory.add(self.origin_pose.rotation, time)
        self.base_angular_motion.set_trajectory(rotation_trajectory)

        self.base_motions = MotionsVector()
        self.base_motions.append(self.com_motion)
        self.base_motions.append(self.base_linear_motion)
        self.base_motions.append(self.base_angular_motion)

        for motion in self.base_motions:
            motion.trajectory.set_start(t)



        q_final = np.copy(self.estimator.estimated_q)
        success = get_end_of_motion_prioritized(self.robot.pin_robot.model, self.ik_data,
                                                self.friction_cones, self.base_motions, q_final)
        if success:
            print("SUCCCCCCCCCCCCCCC")
        else:
            print("FFFFFFFFFFFFFFFF")
        # self.publish_ik(t, q_final)
        pin.computeGeneralizedGravity(
            self.robot.pin_robot.model, self.ik_data, q_final)
        pos = self.compute_com_xy_with_torque()
        com_trajectory = PieceWiseLinearPosition()
        com_trajectory.add(
            np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        com_trajectory.add(pos, time)
        com_trajectory.set_start(t)
        self.com_motion.set_trajectory(com_trajectory)
        success = get_end_of_motion_prioritized(self.robot.pin_robot.model, self.ik_data,
                                                self.friction_cones, self.base_motions, q_final)
        self.publish_ik(t, q_final)
        com_trajectory = PieceWiseLinearPosition()
        com_trajectory.add(
            np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        com_trajectory.add(pos, time)
        com_trajectory.set_start(t)
        self.com_motion.set_trajectory(com_trajectory)

        base_trajectory = PieceWiseLinearPosition()
        base_pose_goal = self.ik_data.oMf[self.base_index]
        base_trajectory.add(
            np.array([self.origin_pose.actInv(base_pose).translation[2]]), 0.0)
        base_trajectory.add(
            np.array([self.origin_pose.actInv(base_pose_goal).translation[2]]), time)
        base_trajectory.set_start(t)
        self.base_linear_motion.set_trajectory(base_trajectory)

        rotation_trajectory = PieceWiseLinearRotation()
        rotation_trajectory.add(base_pose.rotation, 0)
        rotation_trajectory.add(base_pose_goal.rotation, time)
        rotation_trajectory.set_start(t)
        self.base_angular_motion.set_trajectory(rotation_trajectory)

        # success = get_end_of_motion(self.robot.pin_robot.model, self.ik_data,
        #                             self.next_friction_cones, self.motions, q_final)
        # if success:
        #     print("S2S2S2S2S2S2")
        # else:
        #     print("F2F2F2F2F2F2F")
        # self.publish_ik(t, q_final)
        # pin.computeGeneralizedGravity(
        #     self.robot.pin_robot.model, self.ik_data, q_final)
        # pos = self.compute_com_xy_with_torque()
        # com_trajectory = PieceWiseLinearPosition()
        # com_trajectory.add(
        #     np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        # com_trajectory.add(pos, time)
        # com_trajectory.set_start(t)
        # com_motion.set_trajectory(com_trajectory)

        # self.base_motions = MotionsVector()
        # self.base_motions.append(com_motion)
        # self.base_motions.append(base_linear_motion)
        # self.base_motions.append(base_angular_motion)
        self.num_contacts = len(self.friction_cones)
        self.init_qp()

    def update_eef_trajectory(self, t, q, qv, sensors):
        time = self.sequence[self.phase].duration
        for motion in self.sequence[self.phase].motions:
            self.estimator.un_fix(motion.eef_index)

        self.friction_cones = self.estimator.get_friction_cones(
            self.friction_coefficient, self.num_faces)

        self.eef_motions = MotionsVector()
        self.end_times = {}
        fast_time = time/2
        for motion in self.sequence[self.phase].motions:
            eef_motion = EEFPositionMotion(motion.eef_index, np.array(
                [True, True, True], dtype=bool), pin.SE3.Identity(), 2500, 500)
            eef_trajectory = SplineTrajectory(True)
            position = self.robot.pin_robot.data.oMf[motion.eef_index].translation
            eef_trajectory.add(position, 0)
            radius = 0.018
            
            if (len(motion.positions) == 1):
                end_position = motion.positions[-1] + \
                    radius*motion.rotation[:, 2]
                seventy_five = 0.2 * position + 0.8 * end_position
                seventy_five = seventy_five + 0.03*motion.rotation[:, 2]
                eef_trajectory.add(seventy_five, 0.75 * fast_time)
                eef_trajectory.add(end_position, fast_time)
            else:
                for i, position in enumerate(motion.positions):
                    end_position = position + radius*motion.rotation[:, 2]
                    eef_trajectory.add(end_position, motion.times[i]*fast_time)
            eef_trajectory.set_start(t)
            self.publisher.publish(
                get_trajectory_marker(eef_trajectory,  "eef"))
            eef_motion.set_trajectory(eef_trajectory)
            self.end_times[motion.eef_index] = eef_trajectory.end_time()
            self.eef_motions.append(eef_motion)

        q_final = np.copy(self.estimator.estimated_q)
        self.motions = MotionsVector()
        for bm in self.base_motions:
            self.motions.append(bm)
        for em in self.eef_motions:
            self.motions.append(em)

        success = get_end_of_motion_prioritized(self.robot.pin_robot.model, self.ik_data,
                                                self.friction_cones, self.motions, q_final)
        self.publish_ik(t, q_final)
        com_trajectory = PieceWiseLinearPosition()
        com_trajectory.add(
            np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        com_trajectory.add(np.copy(self.ik_data.com[0][:2]), fast_time)
        com_trajectory.set_start(t)
        self.com_motion.set_trajectory(com_trajectory)

        base_trajectory = PieceWiseLinearPosition()
        base_pose = self.robot.pin_robot.data.oMf[self.base_index]
        base_pose_goal = self.ik_data.oMf[self.base_index]
        base_trajectory.add(
            np.array([self.origin_pose.actInv(base_pose).translation[2]]), 0.0)
        base_trajectory.add(
            np.array([self.origin_pose.actInv(base_pose_goal).translation[2]]), fast_time)
        base_trajectory.set_start(t)
        self.base_linear_motion.set_trajectory(base_trajectory)

        rotation_trajectory = PieceWiseLinearRotation()
        rotation_trajectory.add(base_pose.rotation, 0)
        rotation_trajectory.add(base_pose_goal.rotation, fast_time)
        rotation_trajectory.set_start(t)
        self.base_angular_motion.set_trajectory(rotation_trajectory)
        self.init_qp()

    def publish_ik(self, t,  q_final):
        joint_state = JointState()
        joint_state.header.stamp.sec = math.floor(t)
        joint_state.header.stamp.nanosec = math.floor((t-math.floor(t))*1e9)
        joint_state.name = self.robot.joint_names
        joint_state.position = q_final[7:].tolist()
        self.join_state_pub.publish(joint_state)
        world_T_base = TransformStamped()
        world_T_base.header.stamp = joint_state.header.stamp
        world_T_base.header.frame_id = "world"
        world_T_base.child_frame_id = "ik/base_link"
        world_T_base.transform.translation = ToVector(q_final[0:3])
        world_T_base.transform.rotation.w = q_final[6]
        world_T_base.transform.rotation.x = q_final[3]
        world_T_base.transform.rotation.y = q_final[4]
        world_T_base.transform.rotation.z = q_final[5]
        self.tf_broadcaster.sendTransform(world_T_base)

    def init_qp(self):
        eef = 0 if len(
            self.sequence[self.phase].motions) == 0 else self.sequence[self.phase].motions[0].eef_index
        self.controller = WholeBodyController(
            self.estimator, self.friction_cones, self.max_torque, eef)

    def following_spline(self, t, q, qv, sensors):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.ref_acceleration = self.trajectory(t, 2)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, None, q, qv
        )
        return t >= self.transition_end

    def moving_base(self, t, q, qv, sensors):
        finished = all(
            motion.trajectory.finished for motion in self.base_motions)
        self.control = self.controller.compute(
            t, self.robot.pin_robot.model, self.robot.pin_robot.data, self.estimator, self.base_motions, self.control)
        return finished and (self.phase < len(self.sequence) - 1)

    def moving_eef(self, t, q, qv, sensors):
        self.control = self.controller.compute(
            t, self.robot.pin_robot.model, self.robot.pin_robot.data, self.estimator, self.motions, self.control)
        result = self.check_contacts(t)
        return result

    def check_contacts(self, t):
        contact_existing = True
        for motion in self.sequence[self.phase].motions:
            pose = get_touching_pose(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                motion.eef_index,
                motion.rotation[:, 2])
            placement = get_touching_placement(
                self.robot.pin_robot.model, self.robot.pin_robot.data, motion.eef_index, pose
            )
            joint = self.robot.pin_robot.model.frames[motion.eef_index].parentJoint
            vel = pin.getFrameVelocity(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                joint,
                placement,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ).linear
            jacobian = pin.getFrameJacobian(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                joint,
                placement,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            tau = self.robot.pin_robot.data.nle[6:] - self.control
            f = np.matmul(
                np.linalg.inv(jacobian[0:3, joint + 2: joint + 5]).T,
                tau[joint - 4: joint - 1],
            )
            sigma_t = 0.5 * np.sqrt(2)

            sigma_f = 0.2 * np.sqrt(2)
            sigma_v = 0.2 * np.sqrt(2)
            normal = motion.rotation[:, 2]
            v = np.dot(normal, vel)
            f = np.dot(normal, f)
            p = (
                0.5
                * (1 + erf((t - self.end_times[motion.eef_index]) / sigma_t))
                * (1 - erf(np.abs(v) / sigma_v))
                * 0.5 * (1 + erf((f-0.7) / sigma_f))
            )
            contact_existing = contact_existing and (p > 0.25)
        return contact_existing

    def compute_control(self, t, q, qv, sensors):
        if self.estimator:
            self.estimator.estimate(t, q, qv, sensors)
            full_q = self.estimator.estimated_q
            full_qv = self.estimator.estimated_qv
            self.step = self.step + 1
            pin.centerOfMass(
                self.robot.pin_robot.model, self.robot.pin_robot.data, full_q, full_qv
            )
            pin.crba(self.robot.pin_robot.model,
                     self.robot.pin_robot.data, full_q)
            pin.nonLinearEffects(
                self.robot.pin_robot.model, self.robot.pin_robot.data, full_q, full_qv
            )
            if self.step % 50 == 0:
                marker_array = MarkerArray()
                marker = Marker()
                marker.header.frame_id = "world"
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.ns = "com"
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02
                marker.pose.position = ToPoint(
                    self.robot.pin_robot.data.com[0])
                marker.pose.position.z = 0.0
                marker.id = 100
                marker_array.markers.append(marker)

                self.publisher.publish(marker_array)

        self.tick(t, q, qv, sensors)
        return self.control
