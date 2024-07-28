import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from .conversions import ToPoint
from copy import deepcopy


class SplineData:
    def __init__(self, yaml_config, num_joints, poses) -> None:
        self.durations = np.array(yaml_config["durations"])
        self.poses = np.ndarray(
            (len(self.durations), num_joints), dtype=np.float64)
        for i, pose_name in enumerate(yaml_config["poses"]):
            self.poses[i, :] = poses[pose_name]


def poly5(t, t_start, t_end, logger=None):
    delta_t = t_end - t_start
    tau = (t - t_start)/delta_t
    if logger is not None:
        logger.error("TImes are tstart = {} tend = {} delta_t={} t={} tau={}".format(
            t_start, t_end, delta_t, t, tau))
    s = (((6*tau - 15)*tau)+10)*tau*tau*tau
    sdot = (((30*tau - 60)*tau)+30)*tau*tau/delta_t
    sddot = (((120*tau - 180)*tau)+60)*tau/delta_t/delta_t
    return s, sdot, sddot


class PiecewiseLinear:
    def __init__(self):
        self.loop = False
        self.positions = []
        self.times = []
        self.start = None
        self.finished = False

    def __str__(self) -> str:
        return "Trajecotry with {} points lasing {}".format(len(self.positions), self.times[-1])

    def add(self, position, t):
        self.positions.append(position)
        self.times.append(t)

    def close_loop(self, t):
        self.loop = True
        self.add(self.positions[0], t)

    def set_start(self, t):
        self.start = t

    def get(self, t, logger=None):
        start = self.start if self.start is not None else 0
        if t < start:
            return self.positions[0], 0*self.positions[0], 0*self.positions[0]

        trel = (t-start)
        if self.loop:
            trel = trel % self.times[-1]

        if trel > self.times[-1]:
            self.finished = True
            return self.positions[-1], 0*self.positions[0], 0*self.positions[0]

        segment = 0
        while trel > self.times[segment+1]:
            segment = segment+1

        direction = self.positions[segment+1]-self.positions[segment]
        s, sdot, sddot = poly5(
            trel, self.times[segment], self.times[segment+1], logger)
        return self.positions[segment]+s*direction, sdot * direction, sddot*direction

    def get_trajectory_marker(self, namespace):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.ARROW
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        marker.scale.x = 0.005
        marker.scale.y = 0.0075
        marker.scale.z = 0.01
        times = np.linspace(0, self.times[-1], 20)
        delta_t = times[1] - times[0]
        marker.id = 0
        marker.ns = namespace
        for time in times:
            p, v, _ = self.get(time)
            marker.points.clear()
            marker.points.append(ToPoint(p))
            marker.points.append(ToPoint(p + v*delta_t * 0.5))
            marker_array.markers.append(deepcopy(marker))
            marker.id = marker.id + 1
        return marker_array


def create_square(position, u1, u2, distance, time):
    trajectory = PiecewiseLinear()
    trajectory.add(position, 0)
    trajectory.add(position + u1 * distance, time/4)
    trajectory.add(position + u1 * distance + u2*distance, time/2)
    trajectory.add(position + u2*distance, 3*time/4)
    trajectory.close_loop(time)
    return trajectory
