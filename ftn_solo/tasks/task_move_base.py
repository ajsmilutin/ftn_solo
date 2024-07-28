from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import FeedbackLinearization, PDWithFrictionCompensation
from robot_properties_solo import Solo12Robot
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint, ToQuaternion
from copy import deepcopy
from ftn_solo.utils.trajectories import SplineData, PiecewiseLinear
import proxsuite
from ftn_solo.utils.types import FrictionCone


class Estimator:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.estimated_q = np.zeros(self.robot.pin_robot.nq)
        self.estimated_qv = np.zeros(self.robot.pin_robot.nv)
        self.contacts = dict()
        self.node = Node("estimator")

    def init(self, q, qv, sensors):
        self.estimated_q[2] = 0.0
        self.estimated_q[3:6] = sensors["attitude"][1:4]
        self.estimated_q[6] = sensors["attitude"][0]
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
        self.estimated_q[3:6] = sensors["attitude"][1:4]
        self.estimated_q[6] = sensors["attitude"][0]
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
            if (np.linalg.norm(err) < 0.5e-4):
                break
            step[0:6] = np.dot(np.linalg.pinv(J), err)
            grad = np.linalg.norm(step[0:6])
            self.estimated_q = pin.integrate(
                self.robot.pin_robot.model, self.estimated_q, 0.75*step)

        orientation = pin.Quaternion(self.estimated_q[3:7])
        self.robot.pin_robot.framesForwardKinematics(self.estimated_q)
        J = np.zeros((self.num_contacts*3+3, self.robot.pin_robot.nv))
        alpha = 5
        J[:3, 3:6] = alpha*orientation.matrix().T
        i = 0
        for index, pos in self.contacts.items():
            J[3+i*3: 3+(i+1)*3, :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                          self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            i = i+1
        b = np.zeros(self.num_contacts*3+3)
        b[0:3] = alpha*sensors["imu"][0]
        b[3:] = -np.dot(J[3:, 6:], qv)
        self.estimated_qv[0:6] = np.dot(np.linalg.pinv(J[:, :6]), b)


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
        self.estimator = Estimator(self.robot)
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 1)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 4
        self.friction_cones = dict()

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

    def compute_base_trajectory(self, t, q, qv):
        self.base_trajectory = PiecewiseLinear()
        position = self.robot.pin_robot.data.oMf[self.base_index].translation
        dist = 0.1
        time = 3
        self.base_trajectory.add(deepcopy(position), 0)
        self.base_trajectory.add(
            position + np.array([1.0, 0.0, 0.0])*dist, time)
        self.base_trajectory.add(
            position + np.array([0.0, 0.5, 0.0])*dist, time*2)
        self.base_trajectory.add(
            position + np.array([-1.0, 0.0, 0.0])*dist, time*3)
        self.base_trajectory.add(
            position + np.array([0.0, -0.5, 0.0])*dist, time*4)
        self.base_trajectory.add(
            position + np.array([0.0, 0.0, 1.0])*dist, time*5)
        self.base_trajectory.add(
            position + np.array([0.0, 0.0, -1.0])*dist, time*6)
        self.base_trajectory.close_loop(7*time)
        self.publisher.publish(
            self.base_trajectory.get_trajectory_marker("base_link"))
        self.base_trajectory.set_start(t)
        self.qp = proxsuite.proxqp.dense.QP(
            self.robot.pin_robot.nv*2-6+4*3, self.robot.pin_robot.nv+4*3, 4*self.num_faces, False, proxsuite.proxqp.dense.HessianType.Diagonal)

        for index in self.robot.end_eff_ids:
            self.friction_cones[index] = FrictionCone(0.8, self.num_faces, self.robot.pin_robot.data.oMf[index].translation, np.eye(
                3))
            markers = self.friction_cones[index].get_markers(
                "eef_{}".format(index), show_dual=True)
            self.publisher.publish(markers)

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
        accel = np.zeros(3*len(self.robot.end_eff_ids))
        J = np.zeros((3*len(self.robot.end_eff_ids), nv))
        i = 0
        for index in self.robot.end_eff_ids:
            accel[i*3:(i+1)*3] = pin.getFrameAcceleration(
                self.robot.pin_robot.model, self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            J[i*3: (i+1)*3, :] = pin.getFrameJacobian(self.robot.pin_robot.model,
                                                      self.robot.pin_robot.data, index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            i = i+1

        Hessian = 1e-3*np.eye((dim))
        Hessian[:6, :6] = np.eye(6)
        g = np.zeros(dim)
        g[:3] = -ades
        g[3:6] = -alphades
        A = np.zeros((n_eq, dim))
        A[:nv, :nv] = self.robot.pin_robot.data.M
        A[6:nv, nv:2*nv-6] = -np.eye(nv-6)
        A[:nv, 2*nv-6:] = - J.T
        b = np.zeros((nv+4*3))
        b[:nv] = -self.robot.pin_robot.data.nle
        A[nv:nv+4*3, :nv] = J
        b[nv:] = -accel
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
        return False

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
