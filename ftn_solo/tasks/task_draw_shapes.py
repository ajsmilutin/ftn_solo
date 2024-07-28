from transitions import Machine
import numpy as np
from scipy.interpolate import CubicSpline
from .task_base import TaskBase
from ftn_solo.controllers import FeedbackLinearization
from robot_properties_solo import Solo12Robot
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint
from copy import deepcopy
from sensor_msgs.msg import JointState
from ftn_solo.utils.trajectories import SplineData, create_square


class TaskDrawShapes(TaskBase):
    states = ["start", "draw_shapes"]

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        self.step = 0
        super().__init__(num_joints, robot_type, config_yaml)
        if robot_type == "solo12":
            self.robot = Solo12Robot()
        else:
            raise ("Only solo12 supported")
        self.joint_controller = FeedbackLinearization(
            self.robot.pin_robot, self.config["feedback_linearization"])
        self.cartesian_cotnroller = FeedbackLinearization(
            self.robot.pin_robot, self.config["cartesian_controller"])
        self.parse_poses(self.config["poses"])
        self.on_start = SplineData(
            self.config["on_start"], self.num_joints, self.poses)
        self.loop = []
        for point in self.config["loop"]:
            self.loop.append(SplineData(point, self.num_joints, self.poses))
        self.loop_phase = 0
        self.machine = Machine(
            model=self, states=TaskDrawShapes.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "draw_shapes", conditions="following_spline")
        self.machine.add_transition(
            "tick", "draw_shapes", "draw_shapes", conditions="drawing_shapes")
        self.machine.on_enter_draw_shapes(self.compute_shapes)
        self.shapes = dict()
        self.node = Node("draw_shapes_node")
        self.publisher = self.node.create_publisher(
            MarkerArray, "shape_markers", 1)
        self.ref_publisher = self.node.create_publisher(
            JointState, "ref_state", 1)
        self.diff_publisher = self.node.create_publisher(
            JointState, "difff_state", 1)
        self.old_pos = None
        self.old_vel = None

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

    def publish_shape_markers(self):
        for frame, trajectory in self.shapes.items():
            self.publisher.publish(trajectory.get_trajectory_marker("frame_" + str(frame)))

    def publish_joint_states(self, pos, vel, acc):
        joint_state = JointState()
        joint_state.name = ["x", "y", "z"]
        joint_state.header.stamp = self.node.get_clock().now().to_msg()
        if self.old_pos is not None:
            joint_state.position = pos.tolist()
            joint_state.velocity = ((pos - self.old_pos)/self.dt).tolist()
            joint_state.effort = ((vel - self.old_vel)/self.dt).tolist()
            self.diff_publisher.publish(joint_state)

        joint_state.velocity = deepcopy(vel.tolist())
        joint_state.effort = deepcopy(acc.tolist())
        self.ref_publisher.publish(joint_state)
        self.old_pos = pos
        self.old_vel = vel

    def compute_shapes(self, t, q, qv):
        self.shapes[self.robot.fl_index] = create_square(deepcopy(
            self.robot.pin_robot.data.oMf[self.robot.fl_index].translation), np.array([1, 0, 0]), np.array([0, 1, 0]), 0.10, 10)
        self.shapes[self.robot.fr_index] = create_square(deepcopy(
            self.robot.pin_robot.data.oMf[self.robot.fr_index].translation), np.array([1, 0, 0]), -np.array([0, 1, 0]), 0.05, 10)
        self.shapes[self.robot.hl_index] = create_square(deepcopy(
            self.robot.pin_robot.data.oMf[self.robot.hl_index].translation), np.array([0, 0, 1]), np.array([0, 1, 0]), 0.075, 10)
        self.shapes[self.robot.hr_index] = create_square(deepcopy(
            self.robot.pin_robot.data.oMf[self.robot.hr_index].translation), np.array([0, 0, 1]), -np.array([0, 1, 0]), 0.05, 10)
        self.publish_shape_markers()
        for _, trajectory in self.shapes.items():
            trajectory.set_start(t)

    def following_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.ref_acceleration = self.trajectory(t, 2)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, self.ref_acceleration, q, qv)
        return t >= self.transition_end

    def drawing_shapes(self, t, q, qv):
        J = np.zeros((0, self.robot.pin_robot.nv-6), dtype=np.float64)
        Ades = np.zeros((0, 1), dtype=np.float64)
        Kp = 1000
        Kd = 200
        for frame, trajectory in self.shapes.items():
            pos, vel, acc = trajectory.get(t)
            J_real = pin.getFrameJacobian(
                self.robot.pin_robot.model, self.robot.pin_robot.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            vel_real = pin.getFrameVelocity(
                self.robot.pin_robot.model, self.robot.pin_robot.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            A_real = pin.getFrameAcceleration(
                self.robot.pin_robot.model, self.robot.pin_robot.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            qdddes = acc + Kp * \
                (pos - self.robot.pin_robot.data.oMf[frame].translation) + Kd*(
                    vel - vel_real.linear)
            J = np.vstack((J, J_real[0:3, 6:]))
            Ades = np.vstack((Ades, (qdddes - A_real.linear)[:, None]))
        qdes = np.linalg.solve(J, Ades)
        qdes = qdes.ravel()
        self.control = self.cartesian_cotnroller.compute_control(
            0*qdes, 0*qdes, qdes, 0*qdes, 0*qdes)
        return False

    def compute_control(self, t, q, qv, sensors):
        self.step = self.step+1
        full_q = np.zeros(self.robot.pin_robot.nq)
        full_qv = np.zeros(self.robot.pin_robot.nv)
        full_q[2] = 0.4
        full_q[3:6] = sensors["attitude"][1:4]
        full_q[6] = sensors["attitude"][0]
        full_q[7:] = q
        full_qv[6:] = qv
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
