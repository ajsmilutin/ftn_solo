import numpy as np


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
