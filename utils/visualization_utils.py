import mujoco as mj
import numpy as np


def draw_arrow(scene, pos, dir, radius, color=[1, 0.2, 0.2, 1]):
    scene.ngeom += 1
    mj.mjv_initGeom(scene.geoms[scene.ngeom-1],
                    mj.mjtGeom.mjGEOM_ARROW,  [
                        radius, radius, np.linalg.norm(dir)],
                    pos, np.eye(3, 3, dtype=np.float64).flatten(), color)
    pos2 = pos + dir
    mj.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                         mj.mjtGeom.mjGEOM_ARROW, radius, pos[0], pos[1], pos[2], pos2[0], pos2[1], pos2[2])


def draw_frame(scene, pos, rot, radius, length):
    draw_arrow(scene, pos, rot[:, 0]*length, radius, [1, 0.2, 0.2, 1])
    draw_arrow(scene, pos, rot[:, 1]*length, radius, [0.2, 1.0, 0.2, 1])
    draw_arrow(scene, pos, rot[:, 2]*length, radius, [0.2, 0.2, 1, 1])


def draw_surface(scene, pos, rot, length, color=[1, 1, 0.2, 1]):
    scene.ngeom += 1
    mj.mjv_initGeom(scene.geoms[scene.ngeom-1],
                    mj.mjtGeom.mjGEOM_PLANE, [length, length, 0.002], pos, rot.flatten(), color)


def draw_line(scene, start_point, end_point, width=2, color=[0.2, 0.2, 1, 1]):
    scene.ngeom += 1
    start_point = start_point.flatten()
    end_point = end_point.flatten()
    dist = np.linalg.norm(start_point - end_point)
    mj.mjv_initGeom(scene.geoms[scene.ngeom-1], mj.mjtGeom.mjGEOM_LINE, [dist, dist, dist],
                    start_point, np.eye(3, 3, dtype=np.float64).flatten(), color)
    mj.mjv_connector(scene.geoms[scene.ngeom-1], mj.mjtGeom.mjGEOM_LINE, width, start_point,
                     end_point)


def draw_lines(scene, points, width=2, color=[0.2, 0.2, 1, 1]):
    for i in range(points.shape[0] - 1):
        draw_line(scene, points[i, :], points[i + 1, :], width, color)
