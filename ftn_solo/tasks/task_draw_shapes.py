from transitions import Machine
import numpy as np
from .task_base import TaskWithInitPose
from ftn_solo.controllers import FeedbackLinearization
import pinocchio as pin
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA, String
from ftn_solo.utils.conversions import ToPoint
from copy import deepcopy
from ftn_solo.utils.trajectories import create_square, get_trajectory_marker


class TaskDrawShapes(TaskWithInitPose):
    states = ["start", "drawing_shapes"]

    def __init__(self, num_joints, robot_type, config_yaml) -> None:
        super().__init__(num_joints, robot_type, config_yaml)
        self.cartesian_cotnroller = FeedbackLinearization(
            self.robot.pin_robot, self.config["cartesian_controller"]
        )
        self.machine = Machine(
            model=self, states=TaskDrawShapes.states, initial="start"
        )
        self.machine.add_transition(
            "tick", "start", "drawing_shapes", conditions="following_spline"
        )
        self.machine.add_transition(
            "tick", "drawing_shapes", "drawing_shapes", conditions="draw_shapes"
        )
        self.machine.on_enter_drawing_shapes(self.compute_shapes)
        self.shapes = dict()
        self.node = Node("draw_shapes_node")
        self.publisher = self.node.create_publisher(MarkerArray, "shape_markers", 1)
        self.status_publisher = self.node.create_publisher(String, "status", 10)

    def publish_shape_markers(self):
        for frame, trajectory in self.shapes.items():
            self.publisher.publish(
                get_trajectory_marker(trajectory, "frame_" + str(frame))
            )

    def compute_shapes(self, t, q, qv):
        self.shapes[self.robot.fl_index] = create_square(
            deepcopy(self.robot.pin_robot.data.oMf[self.robot.fl_index].translation),
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            0.1,
            5,
        )
        self.shapes[self.robot.fr_index] = create_square(
            deepcopy(self.robot.pin_robot.data.oMf[self.robot.fr_index].translation),
            np.array([1, 0, 0]),
            -np.array([0, 1, 0]),
            0.1,
            5,
        )
        self.shapes[self.robot.hl_index] = create_square(
            deepcopy(self.robot.pin_robot.data.oMf[self.robot.hl_index].translation),
            np.array([0, 0, 1]),
            np.array([0, 1, 0]),
            0.075,
            5,
        )
        self.shapes[self.robot.hr_index] = create_square(
            deepcopy(self.robot.pin_robot.data.oMf[self.robot.hr_index].translation),
            np.array([0, 0, 1]),
            -np.array([0, 1, 0]),
            0.075,
            5,
        )
        self.publish_shape_markers()
        for _, trajectory in self.shapes.items():
            trajectory.set_start(t)

    def draw_shapes(self, t, q, qv):
        J = np.zeros((0, self.robot.pin_robot.nv - 6), dtype=np.float64)
        Ades = np.zeros((0, 1), dtype=np.float64)
        Vdes = np.zeros((0, 1), dtype=np.float64)
        Kp = 400
        Kd = 10
        for frame, trajectory in self.shapes.items():
            pos, vel, acc = trajectory.get(t)
            J_real = pin.getFrameJacobian(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                frame,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            vel_real = pin.getFrameVelocity(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                frame,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            A_real = pin.getFrameAcceleration(
                self.robot.pin_robot.model,
                self.robot.pin_robot.data,
                frame,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            veldes = vel + Kp * (pos - self.robot.pin_robot.data.oMf[frame].translation)
            accdes = acc + Kd * (veldes - vel_real.linear)
            J = np.vstack((J, J_real[0:3, 6:]))
            Vdes = np.vstack((Vdes, veldes[:, None]))
            Ades = np.vstack((Ades, (accdes - A_real.linear)[:, None]))

        qvdes = np.linalg.solve(J, Vdes.ravel())
        qades = np.linalg.solve(J, Ades.ravel())
        self.control = self.cartesian_cotnroller.compute_control(
            0 * qvdes, qvdes, qades, 0 * qades, 0 * qades
        )
        return False

    def compute_control(self, t, q, qv, sensors):
        self.step = self.step + 1
        if not self.estimator.initialized():
            self.estimator.init(t, q, qv, sensors)
        self.estimator.estimate(t, q, qv, sensors)
        self.robot.forward_robot(
            self.estimator.estimated_q, self.estimator.estimated_qv
        )
        pin.crba(
            self.robot.pin_robot.model,
            self.robot.pin_robot.data,
            self.estimator.estimated_q,
        )
        pin.nonLinearEffects(
            self.robot.pin_robot.model,
            self.robot.pin_robot.data,
            self.estimator.estimated_q,
            self.estimator.estimated_qv,
        )
        pin.computeGeneralizedGravity(
            self.robot.pin_robot.model,
            self.robot.pin_robot.data,
            self.estimator.estimated_q,
        )

        if self.step % 50 == 0:
            marker_array = MarkerArray()
            id = 0
            for index in (
                self.robot.hl_index,
                self.robot.hr_index,
                self.robot.fl_index,
                self.robot.fr_index,
            ):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
                marker.scale.x = 0.05
                marker.scale.y = 0.05
                marker.scale.z = 0.05
                marker.pose.position = ToPoint(
                    self.robot.pin_robot.data.oMf[index].translation
                )
                marker.id = id
                id = id + 1
                marker_array.markers.append(marker)

            self.publisher.publish(marker_array)

        self.tick(
            t,
            self.estimator.estimated_q[-self.num_joints :],
            self.estimator.estimated_qv[-self.num_joints :],
        )
        return self.control
