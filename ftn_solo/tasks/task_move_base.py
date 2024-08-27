from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import FeedbackLinearization, PDWithFrictionCompensation
from robot_properties_solo import Solo12Robot
from geometry_msgs.msg import Point, Vector3
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint, ToQuaternion
from copy import deepcopy
from scipy.special import erf, erfc
from ftn_solo.utils.trajectories import get_trajectory_marker, SplineData

import time
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
    EEFLinearMotion,
)
from ftn_solo.utils.wcm import compute_wcm, project_wcm
from dataclasses import dataclass
from ftn_solo.utils.motion import EEFAngularMotion, COMLinearMotion


class MotionData:
    def __init__(self, config):
        self.eef_index = config["eef"]
        position = config["position"] if "position" in config else [
            0.0, 0.0, 0.0]
        quaternion = (
            config["orientation"] if "orientation" in config else [
                0.0, 0.0, 0.0, 1.0]
        )
        self.pose = pin.XYZQUATToSE3(position + quaternion)


class Phase:
    def __init__(self, order, config):
        self.order = order
        self.duration = config["duration"]
        self.motions = []
        for motion_cfg in config["motions"]:
            self.motions.append(MotionData(motion_cfg))


def parse_sequence(config):
    sequence = []
    for i, phase_config in enumerate(config):
        sequence.append(Phase(i, phase_config))
    return sequence


class TaskMoveBase(TaskBase):
    states = ["start", "move_base"]

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
        self.loop = []
        for point in self.config["loop"]:
            self.loop.append(SplineData(point, self.num_joints, self.poses))
        self.loop_phase = 0
        self.machine = Machine(
            model=self, states=TaskMoveBase.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "move_base", conditions="following_spline"
        )
        self.machine.add_transition(
            "tick", "move_base", "move_base", conditions="moving_base"
        )
        self.machine.on_enter_move_base(self.compute_base_trajectory)
        self.node = Node("node")
        self.estimator = None
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 10)
        self.pose_publisher = self.node.create_publisher(
            PoseArray, "origin_pose", 10)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 4
        self.friction_cones = dict()
        self.sequence = parse_sequence(self.config["crawl"])
        self.phase = -1
        self.motions = []
        self.end_times = {}

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
            for motion in self.sequence[self.phase - 1].motions:
                self.estimator.set_fixed(
                    motion.eef_index, motion.pose.rotation)
            if self.phase < len(self.sequence):
                for motion in self.sequence[self.phase].motions:
                    self.estimator.un_fix(motion.eef_index)

        self.friction_cones = self.estimator.get_friction_cones(
            0.8, self.num_faces)

    def publish_wcm_markers(self, wcm_2d, next_wcm_2d, next_com):
        for cone in self.friction_cones:
            publish_cone_marker(cone.data())
        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
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
        for vertex in wcm_2d.vertices:
            point = Point()
            point.z = 0.0
            point.x = wcm_2d.points[vertex, 0]
            point.y = wcm_2d.points[vertex, 1]
            marker.points.append(point)
        marker.points.append(marker.points[0])
        markers.markers.append(marker)
        marker2 = deepcopy(marker)
        marker2.color.g = 1.0
        marker2.id = 206
        marker2.ns = "next_wcm"
        marker2.points.clear()
        for vertex in next_wcm_2d.vertices:
            point = Point()
            point.z = 0.0
            point.x = next_wcm_2d.points[vertex, 0]
            point.y = next_wcm_2d.points[vertex, 1]
            marker2.points.append(point)
        marker2.points.append(marker2.points[0])
        markers.markers.append(marker2)
        marker3 = deepcopy(marker)
        marker3.ns = "com"
        marker3.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        marker3.scale = Vector3(x=0.02, y=0.02, z=0.02)
        marker3.id = 207
        marker3.type = Marker.SPHERE
        marker3.pose.position.x = next_com[0]
        marker3.pose.position.y = next_com[1]
        markers.markers.append(marker3)
        self.publisher.publish(markers)

    def compute_com_xy(self, publish_markers=True):
        wcm = compute_wcm(self.friction_cones)
        wcm_2d = project_wcm(wcm)
        mean_pt = np.mean(wcm_2d.points[wcm_2d.vertices, :], axis=0)
        next_friction_cones = FrictionConeMap()
        for f in self.friction_cones:
            next_friction_cones[f.key()] = f.data()
        if self.phase < len(self.sequence):
            for motion in self.sequence[self.phase].motions:
                next_friction_cones[motion.eef_index] = FrictionCone(
                    0.8, self.num_faces, motion.pose
                )
        if self.phase + 1 < len(self.sequence):
            for motion in self.sequence[self.phase + 1].motions:
                del next_friction_cones[motion.eef_index]
        next_wcm = compute_wcm(next_friction_cones)
        next_wcm_2d = project_wcm(next_wcm)
        C = np.vstack((wcm_2d.equations, next_wcm_2d.equations))
        DD = np.copy(C)
        d = -C[:, 2]
        C[:, 2] = -1
        qp = proxsuite.proxqp.dense.QP(
            3, 0, C.shape[0], False, proxsuite.proxqp.dense.HessianType.Zero
        )
        qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.NO_INITIAL_GUESS
        qp.init(
            np.zeros((3, 3)),
            np.array([0, 0, 1]),
            None,
            None,
            C,
            -1e20 * np.ones(C.shape[0]),
            d,
        )

        qp.solve()
        pos = qp.results.x[:2]
        if publish_markers:
            self.publish_wcm_markers(wcm_2d, next_wcm_2d, pos)

        return pos

    def compute_base_trajectory(self, t, q, qv, sensors):
        self.update_phase(t, q, qv, sensors)
        pos = self.compute_com_xy()

        time = self.sequence[self.phase].duration
        # create base trajectory

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
        origin_pose = pin.SE3(rot, origin)
        poses_msg = PoseArray()
        poses_msg.header.frame_id = "world"
        pose_msg = Pose()
        pose_msg.position = ToPoint(origin_pose.translation)
        pose_msg.orientation = ToQuaternion(
            pin.Quaternion(origin_pose.rotation))
        poses_msg.poses.append(pose_msg)
        self.pose_publisher.publish(poses_msg)

        com_trajectory = PieceWiseLinearPosition()
        com_trajectory.add(np.copy(self.robot.pin_robot.data.com[0][:2]), 0.0)
        com_trajectory.add(pos, time)
        com_motion = COMLinearMotion(
            selected=[True, True, False], Kp=200, Kd=50)
        com_motion.set_trajectory(com_trajectory)

        base_pose = self.robot.pin_robot.data.oMf[self.base_index]
        base_trajectory = PieceWiseLinearPosition()
        base_trajectory.add(np.array([base_pose.translation[2]]), 0.0)
        base_trajectory.add(np.array([0.25]), time)

        base_linear_motion = EEFLinearMotion(
            self.base_index, np.array(
                [False, False, True], dtype=bool), origin_pose, 200, 50
        )
        base_linear_motion.set_trajectory(base_trajectory)

        base_angular_motion = EEFAngularMotion(self.base_index, Kp=200, Kd=50)
        rotation_trajectory = PieceWiseLinearRotation()
        rotation_trajectory.add(base_pose.rotation, 0)
        rotation_trajectory.add(origin_pose.rotation, time)
        base_angular_motion.set_trajectory(rotation_trajectory)

        self.motions = [com_motion, base_linear_motion,
                        base_angular_motion]
        self.motions_dim = 0
        for motion in self.motions:
            motion.trajectory.set_start(t)
            self.motions_dim = self.motions_dim + motion.dim
        # create foot trajectory
        self.end_times = {}
        for motion in self.sequence[self.phase].motions:
            eef_motion = EEFLinearMotion(motion.eef_index, np.array(
                [True, True, True], dtype=bool), pin.SE3.Identity(), 2500, 500)
            eef_trajectory = SplineTrajectory(True)
            position = self.robot.pin_robot.data.oMf[motion.eef_index].translation
            eef_trajectory.add(position, 0)
            radius = 0.018
            end_position = motion.pose.translation + \
                radius*motion.pose.rotation[:, 2]
            seventy_five = 0.25 * position + 0.75 * end_position
            seventy_five[2] = seventy_five[2] + 0.03
            eef_trajectory.add(seventy_five, 0.75 * time)
            eef_trajectory.add(end_position, time)
            eef_trajectory.set_start(t)
            self.publisher.publish(
                get_trajectory_marker(eef_trajectory,  "eef"))
            eef_motion.set_trajectory(eef_trajectory)
            self.end_times[motion.eef_index] = eef_trajectory.end_time()
            self.motions.append(eef_motion)
            self.motions_dim = self.motions_dim + eef_motion.dim

        self.num_contacts = len(self.friction_cones)
        self.init_qp()
        self.qp_times = []

    def init_qp(self):
        self.qp = proxsuite.proxqp.dense.QP(
            self.robot.pin_robot.nv * 2 - 6 + self.num_contacts * 3,
            self.robot.pin_robot.nv + self.num_contacts * 3,
            self.num_contacts * self.num_faces + self.robot.pin_robot.nv - 6,
            False,
            proxsuite.proxqp.dense.HessianType.Dense,
        )
        self.initialized = False

    def following_spline(self, t, q, qv, sensors):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.ref_acceleration = self.trajectory(t, 2)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, None, q, qv
        )
        return t >= self.transition_end

    def moving_base(self, t, q, qv, sensors):
        dim = self.qp.model.dim
        n_eq = self.qp.model.n_eq
        nv = self.robot.pin_robot.nv

        finished = len(
            self.end_times) == 0 and self.motions[0].trajectory.finished
        # estimate contacts
        for contact in self.end_times:
            pose = get_touching_pose(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                contact,
                np.array([0, 0, 1]),
            )
            placement = get_touching_placement(
                self.robot.pin_robot.model, self.robot.pin_robot.data, contact, pose
            )
            joint = self.robot.pin_robot.model.frames[contact].parentJoint
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
                np.linalg.inv(jacobian[0:3, joint + 2: joint + 5]),
                tau[joint - 4: joint - 1],
            )
            sigma_t = 0.1 * np.sqrt(2)

            sigma_f = 1.5 * np.sqrt(2)
            sigma_v = 0.2 * np.sqrt(2)
            v = np.dot(np.array([0, 0, 1]), vel)
            f = np.dot(np.array([0, 0, 1]), f)
            p = (
                0.5
                * (1 + erf(t - self.end_times[contact]) / sigma_t)
                * (1 - erf(np.abs(v) / sigma_v))
                * (erf(np.abs(f) / sigma_f))
            )
            if p > 0.25:
                finished = True

        # constraints
        accel = self.estimator.acceleration
        vel = self.estimator.velocity
        Jc = self.estimator.constraint
        # motions
        Jm = np.zeros((self.motions_dim, dim))
        amdes = np.zeros(self.motions_dim)
        aa = np.zeros(self.motions_dim)
        bb = np.zeros(self.motions_dim)
        start_index = 0
        for motion in self.motions:
            Jm[start_index: start_index + motion.dim, :nv] = motion.get_jacobian(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                self.estimator.estimated_q,
                self.estimator.estimated_qv,
            )
            aa[start_index: start_index + motion.dim] = (
                motion.get_desired_acceleration(
                    t, self.robot.pin_robot.model, self.robot.pin_robot.data
                )
            )
            bb[start_index: start_index + motion.dim] = motion.get_acceleration(
                self.robot.pin_robot.model, self.robot.pin_robot.data
            )
            start_index = start_index + motion.dim

        amdes = aa - bb

        Hessian = np.matmul(Jm.T, Jm)
        g = -np.matmul(Jm.T, amdes)
        A = np.zeros((n_eq, dim))
        A[:nv, :nv] = self.robot.pin_robot.data.M
        A[6:nv, nv: 2 * nv - 6] = -np.eye(nv - 6)
        A[:nv, 2 * nv - 6:] = -Jc.T
        b = np.zeros((nv + self.num_contacts * 3))
        b[:nv] = -self.robot.pin_robot.data.nle
        A[nv: nv + 4 * 3, :nv] = Jc
        b[nv:] = -accel - 20 * vel
        if not self.initialized:
            C = np.zeros((self.num_contacts * self.num_faces + nv - 6, dim))
            start = nv + nv - 6
            for i, cone in enumerate(self.friction_cones):
                C[i * 4: i * 4 + 4, start + i * 3: start + i * 3 + 3] = (
                    cone.data().primal.face
                )
            C[self.num_contacts * self.num_faces:,
                nv: 2 * nv - 6] = np.eye(nv - 6)
            d = 0.25 * np.ones(self.num_contacts * self.num_faces + nv - 6)
            d[self.num_contacts * self.num_faces:] = -2.2
            u = 1e20 * np.ones(self.num_contacts * self.num_faces + nv - 6)
            u[self.num_contacts * self.num_faces:] = 2.2
            self.qp.init(Hessian, g, A, b, C, d, u)
            self.initialized = True
        else:
            self.qp.update(H=Hessian, g=g, A=A, b=b)
        self.qp.solve()
        self.control = self.qp.results.x[nv: 2 * nv - 6]
        return finished and (self.phase < len(self.sequence) - 1)

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
