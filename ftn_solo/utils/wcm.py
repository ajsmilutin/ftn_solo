import numpy as np
from proxsuite import proxqp
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull
import time as tm


def cross_matrix(vector):
    result = np.matrix([[0,          -vector[2],  vector[1]],
                        [vector[2],           0, -vector[0]],
                        [-vector[1],   vector[0],  0]])
    return result


def cross2(u, v):
    return u[0]*v[1] - u[1]*v[0]


def expand(pt0, pt1, qp, v):
    line = pt1 - pt0
    line = line/np.linalg.norm(line)
    normal = np.array([line[1], -line[0]])
    v[-2:] = -normal
    qp.update(g=v)
    x0 = qp.results.x
    x0[-2:] = 0.5*pt1 + 0.5*pt0
    qp.solve(x0, None, None)
    ptmid = qp.results.x[-2:]
    dist = cross2(ptmid-pt0, line)
    if (dist > 0.0025):
        return expand(pt0, ptmid, qp, v) + expand(ptmid, pt1, qp, v)
    else:
        return [pt0]


def expand_2(pt0, pt1, qp, v, lb, ub):
    line = pt1 - pt0
    line = line/np.linalg.norm(line)
    normal = np.array([line[1], -line[0]])
    v[-2:] = -normal
    qp.update(g=v, l_box=lb, u_box=ub)
    x0 = qp.results.x
    x0[-2:] = 0.5*pt1 + 0.5*pt0
    qp.solve(x0, None, None)
    ptmid = qp.results.x[-2:]
    dist = cross2(ptmid-pt0, line)
    if (dist > 0.0025):
        return expand_2(pt0, ptmid, qp, v, lb, ub) + expand_2(ptmid, pt1, qp, v, lb, ub)
    else:
        return [pt0]


def compute_by_expanding_2(friction_cones):
    upsilon = np.hstack(
        [cone.data().primal.span for cone in friction_cones])
    gamma = np.hstack([np.matmul(cross_matrix(cone.data().get_position()), cone.data().primal.span)
                       for cone in friction_cones])
    nalpha = gamma.shape[1]

    qp = proxqp.dense.QP(nalpha + 2, 6, 0, True,
                         proxqp.dense.HessianType.Zero)
    A = np.zeros((6, nalpha+2))
    A[:3, :nalpha] = upsilon
    A[3:6, :nalpha] = gamma
    A[:, -2] = np.array([0, 0, 0, 0, 1, 0])
    A[:, -1] = np.array([0, 0, 0, -1, 0, 0])
    b = np.array([0, 0, 1, 0, 0, 0])
    lb = np.zeros((nalpha+2))
    lb[-2:] = -1e20
    ub = 1e20*np.ones((nalpha + 2))

    dir = np.array([0.0, 1.0])
    v = np.zeros((nalpha+2))
    v[-2:] = -dir
    start = tm.time()
    H = np.zeros((nalpha+2, nalpha+2))
    C = np.zeros((0, nalpha + 2))
    u = np.zeros((0))
    l = np.zeros((0))
    qp.init(H, v, A, b, C, l, u, lb, ub)
    qp.update(l_box=lb, u_box=ub)
    qp.solve()
    pt0 = qp.results.x[-2:]
    v[-2:] = dir
    qp.update(g=v, l_box=lb, u_box=ub)
    qp.solve(qp.results.x, None, None)
    pt1 = qp.results.x[-2:]
    result = expand_2(pt0, pt1, qp, v, lb, ub) + expand_2(pt1, pt0, qp, v, lb, ub)
    return result, tm.time()-start


def compute_by_expanding(friction_cones):
    n_cones = 0
    num_sides = []
    for fc in friction_cones:
        num_sides.append(fc.data().get_num_sides())
        n_cones = n_cones + num_sides[-1]

    num_force = 3*len(friction_cones)
    qp = proxqp.dense.QP(num_force + 2, 6, n_cones, False,
                         proxqp.dense.HessianType.Zero)

    A = np.zeros((6, num_force+2))
    start_row = 0
    for fc in friction_cones:
        A[0:3, start_row:start_row+3] = np.eye(3)
        A[3:6, start_row:start_row+3] = cross_matrix(fc.data().get_position())
        start_row = start_row+3
    A[:, -2] = np.array([0, 0, 0, 0, 1, 0])
    A[:, -1] = np.array([0, 0, 0, -1, 0, 0])
    b = np.array([0, 0, 1, 0, 0, 0])
    C = np.zeros((n_cones, num_force+2))
    start_row = 0
    for i, fc in enumerate(friction_cones):
        C[start_row:start_row+num_sides[i], 3*i:3*i+3] = fc.data().primal.face
        start_row = start_row + num_sides[i]

    dir = np.array([0.0, 1.0])
    v = np.zeros((num_force+2))
    v[-2:] = -dir
    start = tm.time()
    qp.init(np.zeros((num_force+2, num_force+2)), v, A, b,
            C, np.zeros((n_cones)), 1e20*np.ones((n_cones)))
    qp.solve()
    pt0 = qp.results.x[-2:]
    v[-2:] = dir
    qp.update(g=v)
    qp.solve(qp.results.x, None, None)
    pt1 = qp.results.x[-2:]
    result = expand(pt0, pt1, qp, v) + expand(pt1, pt0, qp, v)
    return result, tm.time()-start


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
    start = tm.time()
    hull = ConvexHull(points, qhull_options="Qx")
    surfaces = hull.equations
    return -np.column_stack([surfaces[:, 0:2],
                            surfaces[:, 5],
                            surfaces[:, 2:5]]).dot(block_diag(R.T, R.T))


def project_wcm(wcm):
    wcm_2d = np.column_stack((-wcm[:, 4], wcm[:, 3], wcm[:, 2]))
    (n, m) = wcm_2d.shape
    points = np.zeros((0, 2))
    for i in range(n-1):
        for j in range(i+1, n):
            mat = wcm_2d[[i, j], :]
            if np.linalg.matrix_rank(mat[:2, :2]) == 2:
                pos = np.ravel(np.linalg.solve(mat[:2, :2], -mat[:, 2]))
                p = np.matmul(wcm_2d, np.array([pos[0], pos[1], 1]))
                if np.all(p > -1e-6):
                    points = np.vstack((points, pos))
    return ConvexHull(points)
