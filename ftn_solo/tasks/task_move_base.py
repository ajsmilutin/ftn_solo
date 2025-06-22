from transitions import Machine
import numpy as np
from .task_base import TaskWithInitPose
from ftn_solo.controllers import PDWithFrictionCompensation
from robot_properties_solo import Solo12Robot
import pinocchio as pin
from rclpy.node import Node
from rclpy import spin_once
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import ColorRGBA, String
from ftn_solo.utils.conversions import ToPoint, ToQuaternion
from scipy.special import erf
from ftn_solo.utils.trajectories import get_trajectory_marker, SplineData
from sensor_msgs.msg import Joy
from tf2_ros import TransformBroadcaster
from threading import Thread
from ftn_solo_control import (
    FixedPointsEstimator,
    FrictionConeMap,
    SplineTrajectory,
    get_touching_pose,
    get_touching_placement,
    EEFPositionMotion,
    JointMotion,
    WholeBodyController,
    MotionsVector,
    TrajectoryPlanner
)
from copy import deepcopy
import yaml

SQUARE = 2
EX = 0
TRIANGLE = 3
CIRCLE = 1

class MotionData:
    def __init__(self, config):
        self.positions = []
        if "positions" in config:
            for position in config["positions"]:
                self.positions.append(np.array(position))
            self.times = config["times"]
        elif "position" in config:
            self.positions = [np.array(config["position"])]
            if "times" in config:
                self.times = config["times"]
            else:
                self.times = [1]


class EEFMotionData(MotionData):
    def __init__(self, config):
        super().__init__(config)
        self.eef_index = config["eef"]
        quaternion = config["orientation"] if "orientation" in config else [
            0.0, 0.0, 0.0, 1.0]
        self.rotation = pin.XYZQUATToSE3(
            self.positions[0].tolist() + quaternion).rotation


class JointMotionData(MotionData):
    def __init__(self, config):
        super().__init__(config)
        self.joints = config["joints"]


class Phase:
    def __init__(self, order, config):
        self.order = order
        self.duration = config["duration"]
        self.torso_height = None if not "torso_height" in config else config["torso_height"]
        self.motions = []
        for motion_cfg in config["motions"]:
            if "eef" in motion_cfg:
                self.motions.append(EEFMotionData(motion_cfg))
            else:
                self.motions.append(JointMotionData(motion_cfg))


def parse_sequence(config):
    sequence = []
    for i, phase_config in enumerate(config):
        sequence.append(Phase(i, phase_config))
    return sequence


class PlaybackControl:
    states = ["stopped", "running", "one_step"]

    def __init__(self):
        self.machine = Machine(
            model=self, states=PlaybackControl.states, initial="stopped")
        self.machine.add_transition("start", "stopped", "running")
        self.machine.add_transition("start", "one_step", "running")
        self.machine.add_transition("stop", "running", "stopped")
        self.machine.add_transition("stop", "one_step", "stopped")
        self.machine.add_transition("finish_step", "one_step", "stopped")
        self.machine.add_transition("start_step", "stopped", "one_step")
        self.old_msg = Joy()
        self.old_msg.buttons = [False, False, False, False]

    def can_step(self):
        if self.state in ["running", "one_step"]:
            return True
        else:
            return False

    def step(self):
        if self.state == "one_step":
            self.finish_step()

    def released(self, msg,  button):
        if not msg.buttons[button] and self.old_msg.buttons[button]:
            return True
        return False

    def joy_callback(self, msg):
        if self.released(msg, SQUARE) and self.state in ["running", "one_step"]:
            self.stop()
        if self.released(msg, CIRCLE) and self.state in ["stopped", "one_step"]:
            self.start()
        if self.released(msg, EX) and self.state == "stopped":
            self.start_step()
        if self.released(msg, TRIANGLE):
            exit()
        self.old_msg = msg


class TaskMoveBase(TaskWithInitPose):
    states = ["start", "move_base", "move_eef"]

    def __init__(self, num_joints, robot_type, config_yaml) -> None:
        super().__init__(num_joints, robot_type, config_yaml)
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

        self.machine.on_enter_move_base(self.enter_move_base)
        self.machine.on_enter_move_eef(self.enter_move_eef)
        self.node = Node("node")
        self.estimator = None
        self.trajectory_planner = None
        self.publisher = self.node.create_publisher(MarkerArray, "markers", 10)
        self.status_publisher = self.node.create_publisher(
            String, "status", 10)
        self.pose_publisher = self.node.create_publisher(
            PoseArray, "origin_pose", 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.base_index = self.robot.pin_robot.model.getFrameId("base_link")
        self.initialized = False
        self.num_faces = 12
        self.friction_coefficient = 0.85
        self.torso_height = 0.25
        #self.torso_height = 0.3
        self.friction_cones = dict()
        self.sequence = parse_sequence(self.config["crawl"])
        self.phase = -1
        self.base_motions = MotionsVector()
        self.eef_motions = MotionsVector()
        self.motions = MotionsVector()
        self.end_times = {}
        self.ik_data = pin.Data(self.robot.pin_robot.model)
        self.max_torque = self.config["max_torque"]
        self.playback_controller = PlaybackControl()
        self.node.create_subscription(
            Joy, 'joy', self.playback_controller.joy_callback, 10)
        self.can_step = False
        self.status_publisher.publish(String(data="starting"))
        self.new_contact = 0
        self.ending_contact = 0

    def get_tmp_friction_cones(self):
        self.tmp_friction_cones = self.estimator.get_friction_cones(
            self.friction_coefficient, self.num_faces)
        if self.phase >= 0:
            for motion in self.sequence[self.phase].motions:
                if type(motion) is EEFMotionData:
                    self.tmp_friction_cones[motion.eef_index] = self.estimator.create_friction_cone(
                        motion.eef_index, motion.rotation, self.friction_coefficient, self.num_faces)
        self.tmp_next_friction_cones = FrictionConeMap()
        for f in self.tmp_friction_cones:
            self.tmp_next_friction_cones[f.key()] = f.data()
        if self.phase + 1 < len(self.sequence):
            for motion in self.sequence[self.phase+1].motions:
                if type(motion) is EEFMotionData:
                    del self.tmp_next_friction_cones[motion.eef_index]

    def fix_eef(self):
        self.phase = self.phase + 1
        if self.phase > 0:
            phase = self.phase - 1
            for motion in self.sequence[phase].motions:
                if type(motion) is EEFMotionData:
                    self.estimator.set_fixed(
                        motion.eef_index, motion.rotation)
        self.friction_cones = self.estimator.get_friction_cones(
            self.friction_coefficient, self.num_faces)

    def gen_new_cones(self):
        self.next_friction_cones = FrictionConeMap()
        for f in self.friction_cones:
            self.next_friction_cones[f.key()] = f.data()
        if self.phase < len(self.sequence):
            for motion in self.sequence[self.phase].motions:
                if type(motion) is EEFMotionData:
                    del self.next_friction_cones[motion.eef_index]

    def update_phase(self):
        self.fix_eef()
        self.gen_new_cones()

    def always_false(self, *args, **kwargs):
        return False

    def get_new_origin(self, friction_cones):
        if len(friction_cones) == 4:
            x = np.copy(
                friction_cones[self.robot.end_eff_ids[0]].get_position())
            x = x + \
                friction_cones[self.robot.end_eff_ids[1]].get_position()
            x = x - \
                friction_cones[self.robot.end_eff_ids[2]].get_position()
            x = x - \
                friction_cones[self.robot.end_eff_ids[3]].get_position()
            y = np.copy(
                friction_cones[self.robot.end_eff_ids[0]].get_position())
            y = y - \
                friction_cones[self.robot.end_eff_ids[1]].get_position()
            y = y + \
                friction_cones[self.robot.end_eff_ids[2]].get_position()
            y = y - \
                friction_cones[self.robot.end_eff_ids[3]].get_position()
        else:
            if (
                self.robot.end_eff_ids[0] in friction_cones
                and self.robot.end_eff_ids[2] in friction_cones
            ):
                x = friction_cones[self.robot.end_eff_ids[0]].get_position(
                )
                x = x - \
                    friction_cones[self.robot.end_eff_ids[2]
                                   ].get_position()
            else:
                x = friction_cones[self.robot.end_eff_ids[1]].get_position(
                )
                x = x - \
                    friction_cones[self.robot.end_eff_ids[3]
                                   ].get_position()

            if (
                self.robot.end_eff_ids[0] in friction_cones
                and self.robot.end_eff_ids[1] in friction_cones
            ):
                y = friction_cones[self.robot.end_eff_ids[0]].get_position(
                )
                y = y - \
                    friction_cones[self.robot.end_eff_ids[1]
                                   ].get_position()
            else:
                y = friction_cones[self.robot.end_eff_ids[2]].get_position(
                )
                y = y - \
                    friction_cones[self.robot.end_eff_ids[3]
                                   ].get_position()

        origin = np.zeros(3)
        for cone in friction_cones:
            origin = origin + cone.data().get_position()
        origin = origin / len(friction_cones)
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

    def compute_base_trajectory_start(self, t, q, qv, sensors):
        self.get_tmp_friction_cones()
        duration = self.sequence[self.phase].duration              
        self.origin_pose = self.get_new_origin(self.tmp_friction_cones)
        if self.sequence[self.phase].torso_height:
            self.torso_height = self.sequence[self.phase].torso_height
        self.trajectory_planner = TrajectoryPlanner(
            self.robot.pin_robot.model, self.base_index, self.origin_pose)
        self.node.get_logger().error("Torso height: {}".format(self.torso_height))
        self.trajectory_planner.start_computation(t, self.tmp_friction_cones, self.tmp_next_friction_cones,
                                                  self.estimator.estimated_q, duration, self.torso_height, self.max_torque)

    def compute_base_trajectory_finish(self, t):
        self.update_phase()
        self.base_motions = self.trajectory_planner.motions()
        for m in self.base_motions:
            m.set_start(t)
        self.num_contacts = len(self.friction_cones)
        self.init_qp()

    def update_eef_trajectory_start(self, t, q, qv, sensors):
        duration = self.sequence[self.phase].duration
        self.eef_motions = MotionsVector()
        self.joint_motions = MotionsVector()
        self.end_times = {}
        for motion in self.sequence[self.phase].motions:
            if type(motion) is EEFMotionData:
                eef_motion = EEFPositionMotion(motion.eef_index, np.array(
                    [True, True, True], dtype=bool), pin.SE3.Identity(), 300, 8)
                eef_trajectory = SplineTrajectory(True)
                position = self.robot.pin_robot.data.oMf[motion.eef_index].translation
                rotation = self.robot.pin_robot.data.oMf[motion.eef_index].rotation
                eef_trajectory.add(position, 0)
                # radius = 0.016
                # UNITREEEE
                radius = 0.02
                if (len(motion.positions) == 1):
                    #unitree
                    end_position = motion.positions[-1] + \
                        radius*motion.rotation[:, 2] - \
                        0.01*motion.rotation[:, 2]
                    # 0.02
                    twenty_five = 0.8*position  + 0.2* end_position
                    twenty_five = twenty_five + 0.05 * rotation[:,2]
                    seventy_five = 0.2 * position + 0.8 * end_position
                    seventy_five = seventy_five + 0.05 * motion.rotation[:, 2]
                    eef_trajectory.add(twenty_five, 0.25 *
                                       motion.times[0]*duration)
                    eef_trajectory.add(seventy_five, 0.75 *
                                       motion.times[0]*duration)
                    eef_trajectory.add(end_position, motion.times[0]*duration)
                else:
                    for i, position in enumerate(motion.positions):
                        end_position = position + radius*motion.rotation[:, 2]
                        eef_trajectory.add(
                            end_position, motion.times[i]*duration)
                eef_trajectory.set_start(t)
                eef_motion.set_trajectory(eef_trajectory)
                self.end_times[motion.eef_index] = eef_trajectory.end_time()
                self.eef_motions.append(eef_motion)
            else:
                joints = np.array(motion.joints, dtype=np.int32)
                joint_motion = JointMotion(joints, 1000.0, 100.0)
                joint_trajectory = SplineTrajectory(False)
                joint_trajectory.add(self.estimator.estimated_q[joints+7], 0)
                for i, position in enumerate(motion.positions):
                    joint_trajectory.add(position, motion.times[i]*duration)
                joint_trajectory.set_start(t)
                joint_motion.set_trajectory(joint_trajectory)
                self.joint_motions.append(joint_motion)
        self.trajectory_planner.update_eef_trajectory(t, self.tmp_next_friction_cones,
                                                      self.eef_motions, self.joint_motions, self.estimator.estimated_q, duration)

    def update_eef_trajectory_end(self, t):
        for motion in self.sequence[self.phase].motions:
            if type(motion) is EEFMotionData:
                self.estimator.un_fix(motion.eef_index)
        self.friction_cones = self.estimator.get_friction_cones(
            self.friction_coefficient, self.num_faces)
        self.motions = self.trajectory_planner.motions()
        for m in self.motions:
            m.set_start(t)
        self.init_qp()
        self.trajectory_planner = None

    def init_qp(self):
        self.controller = WholeBodyController(
            self.estimator, self.friction_cones, self.max_torque, yaml.dump(self.config["whole_body"]))

    def following_spline(self, t, q, qv, sensors):
        self.ref_position, self.ref_velocity,  self.ref_acceleration = self.trajectory.get(
            t)
        self.control = self.joint_controller.compute_control(
            self.ref_position, self.ref_velocity, None, q, qv
        )
        if (t >= self.transition_end):
            if not self.can_step:
                self.status_publisher.publish(String(data="Waiting to resume"))
                if self.playback_controller.can_step():
                    self.can_step = True
                    self.playback_controller.step()
            elif not self.estimator:
                marker_array = MarkerArray()
                marker = Marker()
                marker.header.frame_id = "world"
                marker.action = Marker.ADD
                marker.type = Marker.CUBE
                marker.ns = "surface"
                print("marker")
                marker.color = ColorRGBA(r=210.0/255.0, g=249.0/255.0, b=219.0/255.0, a=1.0)
                print("color")
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 0.01
                marker.pose.position.x = 0.0
                marker.pose.position.y = 0.0
                marker.pose.position.z = -0.005
                marker.id = 150
                marker_array.markers.append(marker)
                marker2 = deepcopy(marker)
                marker2.scale.x = 0.6
                marker2.scale.y = 1.0
                marker2.scale.z = 0.6
                marker2.pose.position.x = 0.75
                marker2.pose.position.y = 0.0
                marker2.pose.position.z = 0.3
                marker2.pose.orientation.x = 0.0
                marker2.pose.orientation.y = 0.0
                marker2.pose.orientation.z = 0.0
                marker2.pose.orientation.w = 1.0
                marker2.id = 151
                marker_array.markers.append(marker2)
                self.publisher.publish(marker_array)
                self.status_publisher.publish(
                    String(data="Creating estimator"))
                self.estimator = FixedPointsEstimator(
                    0.0010001,
                    self.robot.pin_robot.model,
                    self.robot.pin_robot.data,
                    self.robot.end_eff_ids,
                )
                self.estimator.init(t, q, qv, sensors)
                return False
            elif self.estimator.initialized():
                if not self.trajectory_planner or not self.trajectory_planner.computation_started():
                    self.status_publisher.publish(
                        String(data="Starting initialization"))
                    self.compute_base_trajectory_start(t, q, qv, sensors)
                    return False
                elif self.trajectory_planner.computation_done():
                    self.ending_contact = 0
                    if self.phase + 1 < len(self.sequence):
                        for motion in self.sequence[self.phase+1].motions:
                            if type(motion) is EEFMotionData:
                                self.ending_contact = motion.eef_index
                                break
                    self.status_publisher.publish(
                        String(data="Finished trajectory planning"))
                    self.compute_base_trajectory_finish(t)
                    self.finished = False
                    self.can_step = False
                    return True
        else:
            return False

    def enter_move_base(self, t, q, qv, sensors):
        self.status_publisher.publish(
            String(data="Moving base, phase: {}".format(self.phase)))

    def enter_move_eef(self, t, q, qv, sensors):
        self.status_publisher.publish(
            String(data="Moving eef, phase: {}".format(self.phase)))

    def moving_base(self, t, q, qv, sensors):
        finished = all(
            motion.finished() for motion in self.base_motions)

        for m in self.base_motions:
            alpha = m.get_alpha(t)

    
        self.control = self.controller.compute(
            t, self.robot.pin_robot.model, self.robot.pin_robot.data, self.estimator, self.base_motions,  np.copy(self.control), alpha, self.new_contact, self.ending_contact)

        if not self.finished:
            self.finished = finished and (self.phase < len(self.sequence) - 1)
            return False
        elif not self.can_step:
            self.status_publisher.publish(
                String(data="Waiting to resume, phase: {}".format(self.phase)))
            if self.playback_controller.can_step():
                self.can_step = True
                self.playback_controller.step()
        elif not self.trajectory_planner.update_started():
            self.status_publisher.publish(
                String(data="Starting motion update, phase: {}".format(self.phase)))
            self.update_eef_trajectory_start(t, q, qv, sensors)
            return False
        elif self.trajectory_planner.update_done():
            self.status_publisher.publish(
                String(data="Finishing motion update, phase: {}".format(self.phase)))
            self.update_eef_trajectory_end(t)
            self.finished = False
            self.can_step = False
            self.status_publisher.publish(
                String(data="Moving EEF, phase: {}".format(self.phase)))
            self.new_contact = self.ending_contact
            self.ending_contact = 0
            return True
        return False

    def moving_eef(self, t, q, qv, sensors):
        for m in self.base_motions:
            alpha = m.get_alpha(t)

        self.control = self.controller.compute(
            t, self.robot.pin_robot.model, self.robot.pin_robot.data, self.estimator, self.motions, np.copy(self.control), alpha,  self.new_contact, self.ending_contact)
        if not self.finished:
            contacts = self.check_contacts(t)
            joints = self.check_joints(t)
            self.finished = contacts and joints
            return False
        else:
            if not self.trajectory_planner or not self.trajectory_planner.computation_started():
                self.status_publisher.publish(
                    String(data="EEF starting comutation done, phase: {}".format(self.phase)))
                self.get_tmp_friction_cones()
                duration = self.sequence[self.phase].duration
                self.origin_pose = self.get_new_origin(self.tmp_friction_cones)
                if self.sequence[self.phase].torso_height:
                    self.torso_height = self.sequence[self.phase].torso_height
                self.trajectory_planner = TrajectoryPlanner(
                    self.robot.pin_robot.model, self.base_index, self.origin_pose)
                self.trajectory_planner.start_computation(t, self.tmp_friction_cones, self.tmp_next_friction_cones,
                                                          self.estimator.estimated_q, duration, self.torso_height, self.max_torque)
                self.fix_eef()
                del self.motions[-1]
                self.init_qp()
                return False
            elif not self.can_step:
                self.status_publisher.publish(
                    String(data="Waiting to resume, phase: {}".format(self.phase)))
                if self.playback_controller.can_step():
                    self.can_step = True
                    self.playback_controller.step()
                return False
            elif self.trajectory_planner.computation_done():
                self.status_publisher.publish(
                    String(data="EEF computation done , phase: {}".format(self.phase)))
                self.gen_new_cones()
                self.base_motions = self.trajectory_planner.motions()
                for m in self.base_motions:
                    m.set_start(t)
                self.num_contacts = len(self.friction_cones)
                if self.phase < len(self.sequence):
                    for motion in self.sequence[self.phase].motions:
                        if type(motion) is EEFMotionData:
                            self.ending_contact = motion.eef_index
                            break
                self.init_qp()
                self.finished = False
                self.can_step = False
                return True
        return False

    def check_joints(self, t):
        joints_finished = True
        for motion in self.joint_motions:
            joints_finished = joints_finished and (
                t > motion.trajectory.end_time())
        return joints_finished

    def check_contacts(self, t):
        contact_existing = True
        for motion in self.sequence[self.phase].motions:
            if not (type(motion) is EEFMotionData):
                continue
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
            # unitree
            # t- 0.02
            p = (
                0.5
                * (1 + erf((t - 0.2 -  self.end_times[motion.eef_index]) / sigma_t))
                * (1 - erf(np.abs(v) / sigma_v))
                * 0.5 * (1 + erf((f-0.7) / sigma_f))
            )
            contact_existing = contact_existing and (p > 0.25)
        return contact_existing

    def compute_control(self, t, q, qv, sensors):
        if self.estimator and self.estimator.initialized():
            self.estimator.estimate(t, q, qv, sensors)
            self.step = self.step + 1
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
        spin_once(self.node, timeout_sec=0)   
        return self.control
