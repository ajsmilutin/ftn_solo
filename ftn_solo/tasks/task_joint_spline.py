from transitions import Machine
import numpy as np
from .task_base import TaskWithInitPose

import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from ftn_solo.utils.conversions import ToPoint
from ftn_solo.utils.trajectories import SplineData
from ftn_solo.controllers import PDWithFrictionAndGravityCompensation


class TaskJointSpline(TaskWithInitPose):
    states = ["start", "follow_spline"]

    def __init__(self,  num_joints, robot_type,  config_yaml) -> None:
        self.step = 0
        super().__init__(num_joints, robot_type, config_yaml)
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

        self.joint_controller = PDWithFrictionAndGravityCompensation(
            self.robot.pin_robot, self.config["joint_controller"])
        self.machine.on_enter_follow_spline(self.compute_spline)
        self.node = Node("node")
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 1)

    def compute_spline(self, t, q, qv):
        self.compute_trajectory(t, self.last_pose, self.loop[self.loop_phase])
        self.loop_phase = (self.loop_phase+1) % len(self.loop)

    def compute_control(self, t, q, qv, sensors):
        if not self.estimator.initialized():
            self.estimator.init(t, q, qv, sensors)
        self.estimator.estimate(t, q, qv, sensors)
        pin.crba(self.robot.pin_robot.model, self.robot.pin_robot.data,
                 self.estimator.estimated_q)
        pin.nonLinearEffects(self.robot.pin_robot.model,
                             self.robot.pin_robot.data,  self.estimator.estimated_q, self.estimator.estimated_qv)

        print(self.estimator.estimated_q.shape)
        self.tick(t, self.estimator.estimated_q[-self.num_joints:],
                  self.estimator.estimated_qv[-self.num_joints:])
        return self.control
