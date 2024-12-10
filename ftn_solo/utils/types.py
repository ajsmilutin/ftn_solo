import numpy as np
import pinocchio as pin
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from copy import deepcopy
from .conversions import ToPoint


class Plane():
    def __init__(self):
        self.position = np.zeros(shape=(3, 1), dtype=np.float64)
        self.R = np.ndarray((3, 3), dtype=np.float64)

    def init_from_points(self, points, positive_pose):
        self.position = np.mean(points, axis=1, keepdims=True)
        self.R, v, d = np.linalg.svd(points-self.position)
        # make sure it's right handed frame
        self.R[:, 2] = np.cross(self.R[:, 0], self.R[:, 1])
        if (self.distance(positive_pose) < 0):
            self.distance(positive_pose)
            # flip Z by rotating around x by 180
            self.R[1:3, :] = -self.R[1:3, :]

    def distance(self, points):
        return np.matmul(self.R[:, 2].T, points - self.position)

    def transform_to_plane(self, points):
        return np.matmul(self.R.T, points - self.position)


class SimpleCone:
    def __init__(self, num_sides, vector, start_angle, translation) -> None:
        self.translation = deepcopy(translation)
        self.num_sides = num_sides
        self.face = np.zeros((num_sides, 3))
        self.span = np.zeros((3, num_sides))
        angle = 2*np.pi/num_sides
        Rstart = pin.AngleAxis(start_angle, np.array([0, 0, 1])).matrix()
        R0 = pin.AngleAxis(angle/2, np.array([0, 0, 1])).matrix()
        R = pin.AngleAxis(angle, np.array([0, 0, 1])).matrix()
        self.span[:, 0] = np.dot(
            np.dot(R0, Rstart), vector/np.linalg.norm(vector))
        for i in range(1, num_sides):
            self.span[:, i] = np.dot(R, self.span[:, i-1])
        self.face[0, :] = np.cross(
            self.span[:, 0], self.span[:, 1])
        self.face[0, :] = self.face[0, :]/np.linalg.norm(self.face[0, :])
        for i in range(1, num_sides):
            self.face[i, :] = np.dot(R, self.face[i-1, :])

    def rotate(self, R):
        self.span = np.dot(R, self.span)
        self.face = np.dot(R, self.face.T).T

    def get_marker(self, namespace, color=ColorRGBA(r=0.8, g=0.0, b=1.0, a=0.5), size=0.15):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.action = Marker.ADD
        marker.type = Marker.TRIANGLE_LIST
        marker.color = color
        marker.ns = namespace
        marker.pose.position = ToPoint(self.translation)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        id = 200
        for i in range(self.num_sides):
            marker.points.append(ToPoint(0*self.translation))
            marker.points.append(
                ToPoint( size * self.span[:, i]))
            marker.points.append(
                ToPoint(size * self.span[:, (i+1) % self.num_sides]))
            marker.id = id
            id = id + 1
        return marker


class FrictionCone:
    def __init__(self, mu, num_sides=4, translation=np.zeros(3), rotation=np.eye(3)) -> None:
        vector = np.array([mu, 0, 1])
        self.primal = SimpleCone(num_sides, vector, 0, translation)
        self.dual = SimpleCone(
            num_sides, self.primal.face[0, :], np.pi/num_sides, translation)
        self.primal.rotate(rotation)
        self.dual.rotate(rotation)

    def get_markers(self, namespace_prefix, size=0.15, show_dual=False):
        markers = MarkerArray()
        markers.markers.append(self.primal.get_marker(
            namespace_prefix+"_primal", size=size))
        if show_dual:
            markers.markers.append(self.dual.get_marker(
                namespace_prefix+"_dual", size=size, color=ColorRGBA(r=0.2, g=0.9, b=1.0, a=0.5)))
        return markers
