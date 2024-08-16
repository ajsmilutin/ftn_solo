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
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint, ToQuaternion
from copy import deepcopy
from ftn_solo.utils.trajectories import SplineData, PiecewiseLinear, SplineTrajectory
import proxsuite
from ftn_solo_control import FrictionCone, FixedPointsEstimator, publish_cone_marker
from ftn_solo.utils.wcm import compute_wcm, project_wcm
from dataclasses import dataclass


class Estimator:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.estimated_q = np.zeros(self.robot.pin_robot.nq)
        self.estimated_qv = np.zeros(self.robot.pin_robot.nv)
        self.contacts = dict()
        self.node = Node("estimator")

    def init(self, q, qv, sensors):
        self.estimated_q[2] = 0.0
        self.estimated_q[3:6] = sensors.imu_data.attitude[1:4]
        self.estimated_q[6] = sensors.imu_data.attitude[0]
        self.estimated_q[7:] = q
        self.estimated_qv[6:] = qv
        self.robot.pin_robot.framesForwardKinematics(self.estimated_q)
        mean = np.zeros(3, dtype=np.float64)
        for foot in (self.robot.fl_index, self.robot.fr_index, self.robot.hl_index, self.robot.hr_index):
            self.contacts[foot] = deepcopy(
                self.robot.pin_robot.data.oMf[foot].translation)
            mean = mean + self.contacts[foot]
        mean = mean / 4.0
        self.estimated_q[2] = -mean[2]
        for _, pos in self.contacts.items():
            pos[2] = pos[2]-mean[2]
        self.num_contacts = 4

    def estimate(self, t, q, qv, sensors):
        self.estimated_q[3:6] = sensors.imu_data.attitude[1:4]
        self.estimated_q[6] = sensors.imu_data.attitude[0]
        self.estimated_q[7:] = q
        self.estimated_qv[6:] = qv
        err = 1
        grad = 1
        J = np.zeros((self.num_contacts*3, 6))
        err = np.zeros(self.num_contacts*3)
        step = np.zeros(self.robot.pin_robot.nv)
        while grad > 1e-6:
            self.robot.pin_robot.framesForwardKinematics(self.estimated_q)
            i = 0
            for index, pos in self.contacts.items():
                err[i*3: (i+1)*3] = pos - \
                    self.robot.pin_robot.data.oMf[index].translation
                J[i*3: (i+1)*3, :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                          self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :6]
                i = i+1
            step[0:6] = np.dot(np.linalg.pinv(J), err)
            grad = np.linalg.norm(step[0:6])
            self.estimated_q = pin.integrate(
                self.robot.pin_robot.model, self.estimated_q, 0.75*step)
            if (np.linalg.norm(err) < 0.5e-4):
                break

        orientation = pin.Quaternion(self.estimated_q[3:7])
        self.robot.pin_robot.framesForwardKinematics(self.estimated_q)
        J = np.zeros((self.num_contacts*3+3, self.robot.pin_robot.nv))
        alpha = 10
        J[:3, 3:6] = alpha*orientation.matrix().T
        i = 0
        for index, pos in self.contacts.items():
            J[3+i*3: 3+(i+1)*3, :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                          self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            i = i+1
        b = np.zeros(self.num_contacts*3+3)
        b[0:3] = alpha*sensors.imu_data.angular_velocity
        b[3:] = -np.dot(J[3:, 6:], qv)
        self.estimated_qv[0:6] = np.dot(np.linalg.pinv(J[:, :6]), b)


class MotionData:
    def __init__(self, config):
        self.eef_index = config["eef"]
        position = config["position"] if "position" in config else [
            0.0, 0.0, 0.0]
        quaternion = config["orientation"] if "orientation" in config else [
            0.0, 0.0, 0.0, 1.0]
        self.pose = pin.XYZQUATToSE3(position+quaternion)


class Phase():
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

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        self.step = 0
        super().__init__(num_joints, robot_type, config_yaml)
        if robot_type == "solo12":
            self.robot = Solo12Robot()
        else:
            raise ("Only solo12 supported")
        self.joint_controller = PDWithFrictionCompensation(
            self.robot.pin_robot, self.config["joint_controller"])
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
            "tick", "start", "move_base", conditions="following_spline")
        self.machine.add_transition(
            "tick", "move_base", "move_base", conditions="moving_base")
        self.machine.on_enter_move_base(self.compute_base_trajectory)
        self.node = Node("node")
        self.estimator = FixedPointsEstimator(self.robot.pin_robot.model, self.robot.pin_robot.data,             [
            self.robot.hl_index, self.robot.hr_index, self.robot.fl_index,  self.robot.fr_index])
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 1)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 4
        self.friction_cones = dict()
        self.sequence = parse_sequence(self.config["crawl"])
        self.phase = -1
        self.motions = []
        print("SEQ_{}".format(self.sequence))

    def parse_poses(self, poses):
        self.poses = {}
        for pose_name in poses:
            self.poses[pose_name] = np.array(
                poses[pose_name], dtype=np.float64)

    def init_pose(self, q, qv):
        self.compute_trajectory(0, q, self.on_start)

    def compute_trajectory(self, tstart, q,  sequence):
        self.transition_end = tstart + sequence.durations[-1]
        self.trajectory = CubicSpline(np.hstack(
            ([tstart], tstart + sequence.durations)), np.vstack((q, sequence.poses)), bc_type="clamped")
        self.last_pose = sequence.poses[-1, :]

    def apply_motion(self, phase, friction_cones):
        for motion in self.sequence[phase-1].motions:
            friction_cones[motion.eef_index] = FrictionCone(
                0.8, 4, motion.pose)
        for motion in self.sequence[phase].motions:
            friction_cones.pop(motion.eef_index)

    def compute_base_trajectory(self, t, q, qv):
        self.phase = self.phase+1
        if self.phase == 0:
            for index in self.robot.end_eff_ids:
                self.friction_cones[index] = FrictionCone(0.8, self.num_faces, pin.SE3(
                    np.eye(3), self.robot.pin_robot.data.oMf[index].translation))
        else:
            self.apply_motion(self.phase, self.friction_cones)

        for cone in self.friction_cones.values():
            publish_cone_marker(cone)

        self.base_trajectory = PiecewiseLinear()
        position = self.robot.pin_robot.data.oMf[self.base_index].translation

        wcm = compute_wcm(self.friction_cones)
        wcm_2d = project_wcm(wcm)

        next_friction_cones = {}
        for cone in self.friction_cones:
            next_friction_cones[cone] = self.friction_cones[cone]
        self.apply_motion(self.phase+1, next_friction_cones)

        next_wcm = compute_wcm(next_friction_cones)
        next_wcm_2d = project_wcm(next_wcm)
        C = np.vstack((wcm_2d.equations, next_wcm_2d.equations))
        D = np.copy(C)
        d = -C[:, 2]
        C[:, 2] = -1
        print(C)
        qp = proxsuite.proxqp.dense.QP(
            3, 0, C.shape[0], False, proxsuite.proxqp.dense.HessianType.Zero)
        qp.init(np.zeros((3, 3)), np.array(
            [0, 0, 1]), None, None, C, -1e20*np.ones(C.shape[0]), d)
        qp.solve()
        pos = qp.results.x[:2]

        markers = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.LINE_STRIP
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
        marker.scale.x = 0.005
        marker.id = 205
        marker.ns = "wcm"
        marker.pose.position = Point(x=0.0, y=0.0, z=0.0)
        marker.pose.orientation.x = marker.pose.orientation.y = marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        for vertex in wcm_2d.vertices:
            point = Point()
            point.z = 0.0
            point.x = wcm_2d.points[vertex, 0]
            point.y = wcm_2d.points[vertex, 1]
            marker.points.append(point)
        markers.markers.append(marker)
        marker2 = deepcopy(marker)
        marker2.color.g = 1.0
        marker2.id = 206
        marker2.points.clear()
        for vertex in next_wcm_2d.vertices:
            point = Point()
            point.z = 0.0
            point.x = next_wcm_2d.points[vertex, 0]
            point.y = next_wcm_2d.points[vertex, 1]
            marker2.points.append(point)
        markers.markers.append(marker2)
        marker3 = deepcopy(marker)
        marker3.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        marker3.scale = Vector3(x=0.025, y=0.025, z=0.025)
        marker3.id = 207
        marker3.type = Marker.SPHERE
        marker3.pose.position.x = pos[0]
        marker3.pose.position.y = pos[1]
        markers.markers.append(marker3)
        self.publisher.publish(markers)

        time = 2
        # create base trajectory
        self.base_trajectory.add(deepcopy(position), 0)
        pos2 = deepcopy(position)
        pos2[:2] = pos
        self.base_trajectory.add(pos2, time)

        self.publisher.publish(
            self.base_trajectory.get_trajectory_marker("base_link"))
        self.base_trajectory.set_start(t)
        # create foot trajectory
        self.motions = []
        for motion in self.sequence[self.phase].motions:
            eef_motion = SplineTrajectory()
            position = self.robot.pin_robot.data.oMf[motion.eef_index].translation
            eef_motion.add(position, 0)
            end_position = motion.pose.translation
            seventy_five = 0.25*position+0.75*end_position
            seventy_five[2] = seventy_five[2] + 0.05
            eef_motion.add(seventy_five, 0.75*time)
            eef_motion.add(end_position, time)
            self.publisher.publish(eef_motion.get_trajectory_marker("eef"))
            self.motions.append(eef_motion)

        num_contacts = len(self.friction_cones)
        print(num_contacts)
        self.qp = proxsuite.proxqp.dense.QP(
            self.robot.pin_robot.nv*2-6+num_contacts*3, self.robot.pin_robot.nv+num_contacts*3, num_contacts*self.num_faces, False, proxsuite.proxqp.dense.HessianType.Diagonal)

    def following_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.ref_acceleration = self.trajectory(t, 2)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, None, q, qv)
        return t >= self.transition_end

    def moving_base(self, t, q, qv):
        p, v, a = self.base_trajectory.get(t)
        Kp = 100
        Kd = 20
        ades = a + Kp * \
            (p - self.estimator.estimated_q[:3]) + \
            Kd * (v - self.estimator.estimated_qv[:3])
        alphades = -100*pin.log(pin.Quaternion(
            self.estimator.estimated_q[3:7]).matrix()) - 20*self.estimator.estimated_qv[3:6]
        dim = self.qp.model.dim
        n_eq = self.qp.model.n_eq
        nv = self.robot.pin_robot.nv
        num_contacts = len(self.friction_cones)
        accel = np.zeros(3*num_contacts)
        vel = np.zeros(3*num_contacts)
        J = np.zeros((3*num_contacts, nv))
        i = 0
        for index in self.friction_cones:
            vel[i*3:(i+1)*3] = pin.getFrameVelocity(
                self.robot.pin_robot.model, self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            accel[i*3:(i+1)*3] = pin.getFrameAcceleration(
                self.robot.pin_robot.model, self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            J[i*3: (i+1)*3, :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                      self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            i = i+1

        cost = np.zeros((6+3*len(self.motions), dim))
        cost[:6, :nv] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                             self.robot.pin_robot.data, self.base_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        Hessian = 1e-3*np.eye((dim))
        Hessian[:6, :6] = np.eye(6)
        g = np.zeros(dim)
        g[:3] = -ades
        g[3:6] = -alphades
        A = np.zeros((n_eq, dim))
        A[:nv, :nv] = self.robot.pin_robot.data.M
        A[6:nv, nv:2*nv-6] = -np.eye(nv-6)
        A[:nv, 2*nv-6:] = - J.T
        b = np.zeros((nv+num_contacts*3))
        b[:nv] = -self.robot.pin_robot.data.nle
        A[nv:nv+4*3, :nv] = J
        b[nv:] = -accel - 5*vel
        if not self.initialized:
            C = np.zeros((4*self.num_faces, dim))
            start = nv+nv-6
            for i, cone in enumerate(self.friction_cones.values()):
                C[i*4:i*4+4, start+i*3:start+i*3+3] = cone.primal.face
            d = 0.5*np.ones(4*self.num_faces)
            u = 1e20*np.ones(4*self.num_faces)
            self.qp.init(Hessian, g, A, b, C, d, u)
            self.initialized = True
        else:
            self.qp.update(H=Hessian, g=g, A=A, b=b)

        self.qp.solve()
        self.control = self.qp.results.x[nv:2*nv-6]
        return (self.base_trajectory.finished) and (self.phase < len(self.sequence)-1)

    def compute_control(self, t, q, qv, sensors):
        if (self.step == 0):
            self.estimator.init(q, qv, sensors)
        self.estimator.estimate(t, q, qv, sensors)
        full_q = self.estimator.estimated_q
        full_qv = self.estimator.estimated_qv
        self.step = self.step+1
        self.robot.forward_robot(full_q, full_qv)
        pin.crba(self.robot.pin_robot.model, self.robot.pin_robot.data, full_q)
        pin.nonLinearEffects(self.robot.pin_robot.model,
                             self.robot.pin_robot.data, full_q, full_qv)
        if (self.step % 50 == 0):
            marker_array = MarkerArray()
            id = 0
            for index in (self.robot.hl_index, self.robot.hr_index, self.robot.fl_index, self.robot.fr_index):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.pose.position = ToPoint(
                    self.robot.pin_robot.data.oMf[index].translation)
                marker.id = id
                id = id + 1
                marker_array.markers.append(marker)

            self.publisher.publish(marker_array)

        self.tick(t, q, qv)
        return self.control
