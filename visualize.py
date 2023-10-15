import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('solo_model.xml')
d = mujoco.MjData(m)

total_time = 10
def controller(model, data):
    global total_time
    # put the controller here. This function is called inside the simulation.
    data.ctrl[1] = 1.047
    data.ctrl[2] = -1.57
    data.ctrl[4] = 1.047
    data.ctrl[5] = -1.57
    data.ctrl[7] = 1.047
    data.ctrl[8] = -1.57
    data.ctrl[10] = 1.57
    data.ctrl[11] = -2
    data.ctrl = min(data.time, total_time) / total_time * data.ctrl


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  mujoco.set_mjcb_control(controller)

  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)
  
    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)