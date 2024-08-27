import numpy as np

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from .conversions import ToPoint
from copy import deepcopy
from scipy.interpolate import CubicSpline
import pinocchio as pin


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
    times = np.linspace(0, trajectory.duration(), 20)
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


# def poly5(t, t_start, t_end, logger=None):
#     delta_t = t_end - t_start
#     tau = (t - t_start) / delta_t
#     if logger is not None:
#         logger.error(
#             "TImes are tstart = {} tend = {} delta_t={} t={} tau={}".format(
#                 t_start, t_end, delta_t, t, tau
#             )
#         )
#     s = (((6 * tau - 15) * tau) + 10) * tau * tau * tau
#     sdot = (((30 * tau - 60) * tau) + 30) * tau * tau / delta_t
#     sddot = (((120 * tau - 180) * tau) + 60) * tau / delta_t / delta_t
#     return s, sdot, sddot


class Trajectory:
    def __init__(self):
        self.loop = False
        self.positions = []
        self.times = []
        self.start = None
        self.finished = False
        self.end_time = None

    def set_start(self, t):
        self.start = t
        self.end_time = self.start + self.times[-1]

    def add(self, position, t):
        self.positions.append(position)
        self.times.append(t)

    def get(self, t, logger=None):
        start = self.start if self.start is not None else 0
        if t < start:
            return self.start_point()

        trel = t - start
        if self.loop:
            trel = trel % self.times[-1]

        if trel > self.times[-1]:
            self.finished = True
            return self.end_point(trel)

        return self.get_interpolated(trel, logger=logger)


class ConstOrientation(Trajectory):
    def __init__(self, rotation):
        super().__init__()
        self.rotation = rotation
        self.finished = True
        self.times = [0]

    def start_point(self):
        return self.rotation, np.zeros(3), np.zeros(3)

    def end_point(self, trel):
        return self.start_point()

    def get_interpolated(self, tre, logger=None):
        return self.start_point()


# class PiecewiseLinear(Trajectory):

#     def __str__(self) -> str:
#         return "Trajecotry with {} points lasing {}".format(
#             len(self.positions), self.times[-1]
#         )

#     def close_loop(self, t):
#         self.loop = True
#         self.add(self.positions[0], t)

#     def start_point(self):
#         return self.positions[0], 0 * self.positions[0], 0 * self.positions[0]

#     def end_point(self, trel):
#         return self.positions[-1], 0 * self.positions[0], 0 * self.positions[0]

#     def get_interpolated(self, trel, logger=None):
#         segment = 0
#         while trel > self.times[segment + 1]:
#             segment = segment + 1
#         s, sdot, sddot = poly5(
#             trel, self.times[segment], self.times[segment + 1], logger
#         )
#         return self.interpolate(segment, s, sdot, sddot)

#     def interpolate(self, segment, s, sdot, sddot):
#         direction = self.positions[segment + 1] - self.positions[segment]
#         return (
#             self.positions[segment] + s * direction,
#             sdot * direction,
#             sddot * direction,
#         )


# class PiecewiseLinearRotation(PiecewiseLinear):
#     def __str__(self) -> str:
#         return "Trajecotry with {} points lasing {}".format(
#             len(self.positions), self.times[-1]
#         )

#     def start_point(self):
#         return self.positions[0], np.zeros(3), np.zeros(3)

#     def end_point(self, trel):
#         return self.positions[-1], np.zeros(3), np.zeros(3)

#     def interpolate(self, segment, s, sdot, sddot):
#         axang = pin.log(
#             np.matmul(self.positions[segment].T, self.positions[segment + 1])
#         )
#         return (
#             np.matmul(self.positions[segment], pin.exp(s*axang)),
#             sdot * np.matmul(self.positions[segment], axang),
#             sddot * np.matmul(self.positions[segment], axang),
#         )


# class SplineTrajectory(Trajectory):
#     def __init__(self, follow_through=False):
#         self.loop = False
#         self.positions = []
#         self.times = []
#         self.start = None
#         self.finished = False
#         self.follow_through = True

#     def add(self, position, t):
#         super().add(position, t)
#         if len(self.times) >= 2:
#             # Clamped on both ends
#             bc_type = ((1, np.zeros(3)), (1, np.zeros(3)))
#             if self.follow_through:
#                 # Clamped on start, natural at the end
#                 bc_type = ((1, np.zeros(3)), (2, np.zeros(3)))
#             self.spline = CubicSpline(np.hstack(self.times), np.vstack(
#                 self.positions), bc_type=bc_type)

#     def start_point(self):
#         return self.positions[0], 0 * self.positions[0], 0 * self.positions[0]

#     def end_point(self, trel):
#         if self.follow_through:
#             return self.spline(trel), self.spline(trel, 1), self.spline(trel, 2)
#         else:
#             return self.positions[-1], 0 * self.positions[0], 0 * self.positions[0]

#     def get_interpolated(self, trel, logger=None):
#         return self.spline(trel), self.spline(trel, 1), self.spline(trel, 2)


def create_square(position, u1, u2, distance, time):
    trajectory = PiecewiseLinear()
    trajectory.add(position, 0)
    trajectory.add(position + u1 * distance, time / 4)
    trajectory.add(position + u1 * distance + u2 * distance, time / 2)
    trajectory.add(position + u2 * distance, 3 * time / 4)
    trajectory.close_loop(time)
    return trajectory
