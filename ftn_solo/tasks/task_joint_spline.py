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


class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray(
            (len(self.durations), num_joints), dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i, :] = poses[pose_name]


class TaskJointSpline(TaskBase):
    states = ["start", "follow_spline"]

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        self.step = 0
        super().__init__(num_joints, robot_type, config_yaml)
        if robot_type == "solo12":
            self.robot = Solo12Robot()
        else:
            raise ("Only solo12 supported")
        self.joint_controller = FeedbackLinearization(
            self.robot.pin_robot, self.config["joint_controller"])
        self.parse_poses(self.config["poses"])
        self.on_start = SplineData(
            self.config["on_start"], self.num_joints, self.poses)
        self.loop = []
        for point in self.config["loop"]:
            self.loop.append(SplineData(point, self.num_joints, self.poses))
        self.loop_phase = 0
        self.machine = Machine(
            model=self, states=TaskJointSpline.states, initial="start")
        self.machine.add_transition(
            "tick", "start", "follow_spline", conditions="following_spline")
        self.machine.add_transition(
            "tick", "follow_spline", "follow_spline", conditions="following_spline")

        self.machine.on_enter_follow_spline(self.compute_spline)
        self.node = Node("node")
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 1)

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

    def compute_spline(self, t, q, qv):
        self.compute_trajectory(t, self.last_pose, self.loop[self.loop_phase])
        self.loop_phase = (self.loop_phase+1) % len(self.loop)

    def following_spline(self, t, q, qv):
        self.ref_position = self.trajectory(t)
        self.ref_velocity = self.trajectory(t, 1)
        self.ref_acceleration = self.trajectory(t, 2)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, self.ref_acceleration, q, qv)
        return t >= self.transition_end

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
