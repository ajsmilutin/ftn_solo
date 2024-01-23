#!/usr/bin/env python
import time

import numpy as np
import mujoco
import mujoco.viewer
# from controllers.controller import Controller
# from utils.visualization_utils import draw_frame, draw_surface

paused = False
leave = False
# robot_controler = Controller()

def controller(model, data):
    des_pos = np.array(
    [
        0.3,   # FLHAA
        0.7,   # FLHFE
        -1.57,  # FLKFE
        -0.3,  # FRHAA
        0.7,   # FRHFE
        -1.57,  # FRKFE
        0.3,   # FLHAA
        0.7,   # FLHFE
        -1.57,  # FLKFE
        -0.3,  # FRHAA
        0.7,   # FRHFE
        -1.57,  # FRKFE
    ])
    q = data.qpos[7:]
    qv = data.qvel[6:]
    touch_sensors = ["fl", "fr", "hl", "hr"]
    reading = {}
    for sensor in touch_sensors:
        name = sensor + "_touch"
        reading[name] = data.sensor(name).data[0] > 0
    sensor = {"imu": (data.sensor("angular-velocity").data, data.sensor(
        "linear-acceleration").data, data.sensor("magnetometer").data),
        "touch": reading}
    data.ctrl = 1.0 * (des_pos - q) - 0.001 * qv

def update_scene(scn, model, data):
    rot = np.zeros((9,), dtype=np.float64)
    scn.ngeom = 0
    # mujoco.mju_quat2Mat(rot, robot_controler.Q)
    # draw_frame(scn, robot_controler.pos, rot.reshape((3, 3)), 0.005, 0.2)
    # if robot_controler.surface:
    #     draw_surface(scn, robot_controler.surface.position, robot_controler.surface.R, 1)


def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused
    elif keycode == 256:  # ESC
        global leave
        leave = True

# def simulate():
m = mujoco.MjModel.from_xml_path("/home/ajsmilutin/solo_ws/src/robot_properties_solo/src/robot_properties_solo/resources/mjcf/solo12.mjcf")
m.body("base").jntnum = 0
print(dir(m.body("base")))
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d, show_right_ui=False, key_callback=key_callback) as viewer:
    mujoco.set_mjcb_control(controller)

    # Close viewer on ESC keyboard press
    while viewer.is_running():
        if paused:
            continue
        step_start = time.time()
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        update_scene(viewer.user_scn, m, d)
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if leave:
            viewer.close()
        print("STOPPP")
    
        
        
    
# if __name__ == '__main__':
#     simulate()    

