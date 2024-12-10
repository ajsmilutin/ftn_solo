import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from .conversions import ToPoint
from copy import deepcopy
import pinocchio as pin
from ftn_solo_control import PieceWiseLinearPosition


def get_trajectory_marker(trajectory, namespace):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "world"
    marker.action = Marker.ADD
    marker.type = Marker.ARROW
    marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
    marker.scale.x = 0.005
    marker.scale.y = 0.0075
    marker.scale.z = 0.01
    times = np.linspace(0, 1.5*trajectory.duration(), 20)
    delta_t = times[1] - times[0]
    marker.id = 0
    marker.ns = namespace
    for time in times:
        p, v, _ = trajectory.get(trajectory.start_time() + time)
        marker.points.clear()
        marker.points.append(ToPoint(p))
        marker.points.append(ToPoint(p + v * delta_t * 0.5))
        marker_array.markers.append(deepcopy(marker))
        marker.id = marker.id + 1
    return marker_array


class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray(
            (len(self.durations), num_joints), dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i, :] = poses[pose_name]

def create_square(position, u1, u2, distance, time):
    trajectory = PieceWiseLinearPosition()
    trajectory.add(position, 0)
    trajectory.add(position + u1 * distance, time / 4)
    trajectory.add(position + u1 * distance + u2 * distance, time / 2)
    trajectory.add(position + u2 * distance, 3 * time / 4)
    trajectory.close_loop(time)
    return trajectory
