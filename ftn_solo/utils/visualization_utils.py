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
