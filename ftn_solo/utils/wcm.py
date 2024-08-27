import numpy as np
from proxsuite import proxqp
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull

def cross_matrix(vector):
    result = np.matrix([[0,          -vector[2],  vector[1]],
                        [vector[2],           0, -vector[0]],
                        [-vector[1],   vector[0],  0]])
    return result


def compute_wcm(friction_cones):
    n = len(friction_cones)
    upsilon = np.hstack(
        [cone.data().primal.span for cone in friction_cones])
    gamma = np.hstack([np.matmul(cross_matrix(cone.data().get_position()), cone.data().primal.span)
                       for cone in friction_cones])

    iksilon = np.vstack([cone.data().dual.face for cone in friction_cones])
    n = iksilon.shape[0]
    v0 = np.array([0.0, 0.0, 1.0])
    found = True
    if not np.all(np.matmul(iksilon, v0) > -1e-6):
        v0 = np.mean(upsilon, axis=1)
        if not np.all(np.matmul(iksilon, v0) > -1e-6):
            s0 = np.max(-np.matmul(iksilon, v0))
            qp = proxqp.dense.QP(4, 0, n, False, proxqp.dense.HessianType.Zero)
            C = np.vstack(
                np.hstack([upsilon, np.ones(n, 1)]), np.array([0, 0, 0, 1]))
            d = np.zeros(n+1)
            d[-1] = -1
            qp.init(np.zeros(4, 4), np.array([0, 0, 0, 1]), np.zeros(
                0, 4), np.zeros(0), C, d, 1e20*np.ons(n+1))
            qp.solve()
            v0 = qp.results.x[0:3]
            found = qp.results.x[3] < 0

    if not found:
        return None
    v0 = v0/np.linalg.norm(v0)
    e1 = np.cross(np.array([0, 1, 0]), v0)
    e1 = e1/np.linalg.norm(e1)
    R = np.column_stack([e1, np.cross(v0, e1), v0])
    upsilon = np.matmul(R.T, upsilon)
    gamma = np.matmul(R.T, gamma)
    gamma = gamma / upsilon[2, :]
    upsilon = upsilon / upsilon[2, :]
    points = np.vstack([upsilon[0:2, :], gamma]).T
    hull = ConvexHull(points, qhull_options="Qx")
    surfaces = hull.equations
    return -np.column_stack([surfaces[:, 0:2],
                            surfaces[:, 5],
                            surfaces[:, 2:5]]).dot(block_diag(R.T, R.T))
    
def project_wcm(wcm):
    wcm_2d = np.column_stack((-wcm[:, 4], wcm[:, 3], wcm[:, 2]))
    (n, m) = wcm_2d.shape
    points = np.zeros((0,2))
    for i in range(n-1):
        for j in range(i+1, n):
            mat = wcm_2d[[i, j], :]
            if np.linalg.matrix_rank(mat[:2, :2]) == 2:
                pos = np.ravel(np.linalg.solve(mat[:2, :2], -mat[:, 2]))
                p = np.matmul(wcm_2d, np.array([pos[0], pos[1], 1]))
                if np.all(p > -1e-6):
                    points =  np.vstack((points,pos))
    return ConvexHull(points)   