from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import FeedbackLinearization, PDWithFrictionCompensation
from robot_properties_solo import Solo12Robot
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, Float32MultiArray
from ftn_solo.utils.conversions import ToPoint, ToQuaternion
from copy import deepcopy
from ftn_solo.utils.trajectories import SplineData, PiecewiseLinear
import proxsuite
from ftn_solo.utils.types import FrictionCone
import rclpy
from sensor_msgs.msg import Joy
from scipy.linalg import expm
import math

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
        self.estimated_qv[0:6] = np.dot(    np.linalg.pinv(J[:, :6]), b)


class TaskMoveBase(TaskBase):
    states = ["start", "move_base", "move_up", "idle"]

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
        
        self.transition_maker()
        
        self.node = Node("node")
        self.estimator = Estimator(self.robot)
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 1)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 4
        self.friction_cones = dict()
        self.subscriber = self.node.create_subscription(Joy,'joy', self.joy_callback, 10)
        self.des_linear_velocity = np.array([0.0, 0.0, 0.0])
        self.des_angular_velocity = np.array([0.0, 0.0, 0.0])
        self.previous_t = 0
        self.inside = True
        self.distance_from_edge = np.zeros(4)
        self.scale_up_down_buttons = 1.0
        self.scale_left_stick_up_down = 1.0
        self.scale_left_stick_left_right = 1.0
        self.scale_right_stick = 1.0

    def joy_callback(self, msg):
        self.msg = msg
        if (msg.axes[0] > 0.05 or msg.axes[0] < -0.05):
            self.des_linear_velocity[1] = msg.axes[0] * 0.15 * self.scale_left_stick_left_right
        else: self.des_linear_velocity[1] = 0
        if (msg.axes[1] > 0.05 or msg.axes[1] < -0.05):
            self.des_linear_velocity[0] = msg.axes[1] * 0.15 * self.scale_left_stick_up_down
        else: self.des_linear_velocity[0] = 0
        if msg.axes[3] > 0.05 or msg.axes[3] < -0.05:
            self.des_angular_velocity[0] = msg.axes[3] * 0.3 * self.scale_right_stick
        else:
            self.des_angular_velocity[0] = 0
        if msg.axes[4] > 0.05 or msg.axes[4] < -0.05:
            self.des_angular_velocity[1] = -msg.axes[4] * 0.3 * (1 - self.scale_right_stick)
        else:   
            self.des_angular_velocity[1] = 0
        if msg.buttons[6]:
            self.des_angular_velocity[2] = msg.buttons[6] * 0.2
        elif msg.buttons[7]:
            self.des_angular_velocity[2] = -msg.buttons[7] * 0.2
        else: self.des_angular_velocity[2] = 0
        if msg.buttons[4]:
            self.des_linear_velocity[2] = -0.2 * (1 - self.scale_up_down_buttons)#-0.5 * ((1 - 0.15 / self.robot.pin_robot.data.oMf[self.base_index].translation[-1]))
        elif msg.buttons[5]:
            self.des_linear_velocity[2] = 0.2 * self.scale_up_down_buttons
        else: self.des_linear_velocity[2] = 0
    
    def transition_maker(self):
        self.machine.add_transition(
            "tick", "start", "move_base", after="compute_base_trajectory", conditions="following_spline")
        self.machine.add_transition(
            "tick", "move_base", "move_base", conditions="moving_base")

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
        self.qp = proxsuite.proxqp.dense.QP(
            self.robot.pin_robot.nv*2-6+4*3, self.robot.pin_robot.nv+4*3, 4*self.num_faces, False, proxsuite.proxqp.dense.HessianType.Diagonal)
        self.des_q = self.estimator.estimated_q
        self.des_qv = np.zeros(self.robot.pin_robot.nv)

        for index in self.robot.end_eff_ids:
            self.friction_cones[index] = FrictionCone(0.8, self.num_faces, 
                self.robot.pin_robot.data.oMf[index].translation, np.eye(3))
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
        Kp = 100
        Kd = 20
        dt = t - self.previous_t

        self.distance_point_to_edge()

        knee_angles = self.get_knee_angles() #fr, fl, hl, hr
        angle_upper_limit = 0.3
        angle_lower_limit = 0.9

        self.scale = (knee_angles[2] - angle_upper_limit) / (angle_lower_limit - angle_upper_limit)
        if self.des_linear_velocity[0] > 0:
            self.scale_left_stick_up_down = (self.distance_from_edge[0] - 0.05) / (0.2 - 0.05)
        elif self.des_linear_velocity[0] < 0:
            self.scale_left_stick_up_down = (self.distance_from_edge[2] - 0.05) / (0.2 - 0.05)
        
        if self.des_linear_velocity[1] > 0:
            self.scale_left_stick_left_right = (self.distance_from_edge[1] - 0.05) / (0.2 - 0.05)
        elif self.des_linear_velocity[1] < 0:
            self.scale_left_stick_left_right = (self.distance_from_edge[3] - 0.05) / (0.2 - 0.05)

        # self.node.get_logger().info(str(self.distance_from_edge))
        self.des_qv[0:3] = self.des_linear_velocity
        self.des_qv[3:6] = self.des_angular_velocity
        self.des_q = pin.integrate(self.robot.pin_robot.model, self.des_q, self.des_qv * dt)

        des_position = self.des_q[0:3]
        current_position = self.estimator.estimated_q[:3]
        des_orientation = pin.Quaternion(self.des_q[3:7]).matrix()
        current_orientation = pin.Quaternion(self.estimator.estimated_q[3:7]).matrix()

        ades = Kp * (des_position - current_position) + Kd * (self.des_linear_velocity - self.estimator.estimated_qv[:3])
        
        alphades = Kp * (np.matmul(current_orientation, pin.log(
                    np.matmul(current_orientation.T, des_orientation)))) \
                    + Kd*(self.des_angular_velocity - self.estimator.estimated_qv[3:6])

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
        self.previous_t = t
        return True

    def get_knee_angles(self):
        fr_angle = self.robot.pin_robot.data.oMf[self.robot.fr_index].rotation
        fr_angle = np.arctan2(-fr_angle[2, 0], np.sqrt(fr_angle[0, 0]**2 + fr_angle[1, 0]**2))

        hr_angle = self.robot.pin_robot.data.oMf[self.robot.hr_index].rotation
        hr_angle = np.arctan2(-hr_angle[2, 0], np.sqrt(hr_angle[0, 0]**2 + hr_angle[1, 0]**2))

        fl_angle = self.robot.pin_robot.data.oMf[self.robot.fl_index].rotation
        fl_angle = np.arctan2(-fl_angle[2, 0], np.sqrt(fl_angle[0, 0]**2 + fl_angle[1, 0]**2))

        hl_angle = self.robot.pin_robot.data.oMf[self.robot.hl_index].rotation
        hl_angle = np.arctan2(-hl_angle[2, 0], np.sqrt(hl_angle[0, 0]**2 + hl_angle[1, 0]**2))

        return np.array([fr_angle, fl_angle, hl_angle, hr_angle])

    def draw_backbone_verticals(self):
        marker_array = MarkerArray()
        id = 0
        for index in (self.robot.hl_index, self.robot.hr_index, self.robot.fl_index, self.robot.fr_index):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.action = Marker.ADD
            marker.type = Marker.CYLINDER
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
            marker.scale.x = 0.01
            marker.scale.y = 0.01 
            marker.scale.z = 0.50
            marker.pose.position = ToPoint(
                self.robot.pin_robot.data.oMf[index].translation)
            marker.id = id
            id = id + 1
            marker_array.markers.append(marker)
        self.publisher.publish(marker_array)

    def distance_from_polygon(self):
        x, y, z = self.robot.pin_robot.data.com[0]
        n = 4
        inside = True
        
        legs = (self.robot.fr_index, self.robot.fl_index, self.robot.hl_index, self.robot.hr_index)

        for i in range(n):
            x1, y1, z1 = self.robot.pin_robot.data.oMf[legs[i]].translation
            x2, y2, z2 = self.robot.pin_robot.data.oMf[legs[(i + 1) % n]].translation  # Next vertex, wrap around at the end

            # Vector from edge start to edge end
            edge_vector = np.array([x2 - x1, y2 - y1])

            self.distance_from_edge[i] = (edge_vector[0] * x - edge_vector[1] * y) / (np.sqrt(edge_vector[0]**2 + edge_vector[1]**2))
        
        self.node.get_logger().info(str(self.distance_from_edge))

    def is_point_inside_convex_polygon(self):

        x, y, z = self.robot.pin_robot.data.com[0]
        n = 4
        inside = True
        
        legs = (self.robot.fr_index, self.robot.fl_index, self.robot.hl_index, self.robot.hr_index)
        i = 0
        for i in range(n):
            x1, y1, z1 = self.robot.pin_robot.data.oMf[legs[i]].translation
            x2, y2, z2 = self.robot.pin_robot.data.oMf[legs[(i + 1) % n]].translation  # Next vertex, wrap around at the end

            # Vector from edge start to edge end
            edge_vector = np.array([x2 - x1, y2 - y1])
            # Vector from edge start to the point
            point_vector = np.array([x - x2, y - y2])
            
            # Cross product to determine which side the point is on
            cross_product = np.cross(edge_vector, point_vector)

            if cross_product < 0.015:  # Point is outside the polygon if cross product is negative
                inside = False
                break
        
        return inside

    def distance_point_to_edge(self):
        x, y, z = self.robot.pin_robot.data.com[0]
        n = 4
        inside = True
        
        legs = (self.robot.fr_index, self.robot.fl_index, self.robot.hl_index, self.robot.hr_index)
        for i in range(n):
            x1, y1, z1 = self.robot.pin_robot.data.oMf[legs[i]].translation
            x2, y2, z2 = self.robot.pin_robot.data.oMf[legs[(i + 1) % n]].translation

            # Vector from edge start to edge end
            edge_vector = np.array([x2 - x1, y2 - y1])
            # Vector from edge start to the point
            point_vector = np.array([x - x1, y - y1])

            magnitude = edge_vector[0]**2 + edge_vector[1]**2

            if magnitude == 0:
                closest_point = np.array([x1, y1])
            else:
                projection = (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) \
                            / magnitude
                projection = max(0, min(1, projection))

                closest_point = np.array([x1 + projection * edge_vector[0], y1 + projection * edge_vector[1]])

            dx = x - closest_point[0]
            dy = y - closest_point[1]        
            self.distance_from_edge[i] = math.sqrt(dx**2 + dy**2)
        

    def compute_control(self, t, q, qv, sensors):
        if (self.step == 0):
            self.estimator.init(q, qv, sensors)
        self.estimator.estimate(t, q, qv, sensors)

        full_q = self.estimator.estimated_q
        full_qv = self.estimator.estimated_qv
        self.step = self.step+1
        self.robot.forward_robot(full_q, full_qv)

        pin.centerOfMass(self.robot.pin_robot.model, self.robot.pin_robot.data, full_q, full_qv, True)
        com = self.robot.pin_robot.data.vcom[0]
        # todo: make a marker for com and represent it in rviz for better understanding

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

            marker = Marker()
            marker.header.frame_id = "world"
            marker.action = Marker.ADD
            marker.type = Marker.SPHERE
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.1
            marker.pose.position = ToPoint(com)
            marker.id = id
            id = id + 1
            marker_array.markers.append(marker)
            self.publisher.publish(marker_array)

        rclpy.spin_once(self.node, timeout_sec=0)


        self.tick(t, q, qv)
        return self.control