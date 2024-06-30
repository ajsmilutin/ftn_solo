"""wrapper

Pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2021, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
try:
    # use standard Python importlib if available (Python>3.7)
    import importlib.resources as importlib_resources
except ImportError:
    import importlib_resources
import pybullet
import time
from robot_properties_solo.robot_resources import Resources


class BulletEnv(object):
    def __init__(self, robot_version, server=pybullet.GUI, dt=0.001):
        self.dt = dt
        self.objects = []
        self.robots = []

        self.physics_client = pybullet.connect(server)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=dt, numSubSteps=1)
        self.resources = Resources(robot_version)

    def add_object_from_urdf(self, urdf_path, pos=[0, 0, 0], orn=[0, 0, 0, 1], useFixedBase=True):
        object_id = pybullet.loadURDF(urdf_path, useFixedBase=useFixedBase)
        pybullet.resetBasePositionAndOrientation(object_id, pos, orn)
        self.objects.append(object_id)
        return object_id

    def start_video_recording(self, file_name):
        self.file_name = file_name
        pybullet.startStateLogging(
            pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_video_recording(self):
        if hasattr(self, "file_name"):
            pybullet.stopStateLogging(
                pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def step(self, sleep=False):
        if sleep:
            time.sleep(self.dt)
        pybullet.stepSimulation()

    def print_physics_engine_params(self):
        params = pybullet.getPhysicsEngineParameters(self.physicsClient)
        print("physics_engine_params:")
        for key in params:
            print("    - ", key, ": ", params[key])


class BulletEnvWithGround(BulletEnv):
    def __init__(self, robot_version, server=pybullet.GUI, dt=0.001):
        super().__init__(robot_version, server, dt)
        plane_urdf = self.resources.urdf_plane_path
        self.add_object_from_urdf(plane_urdf)

    def set_floor_frictions(self, lateral=1.0, spinning=0.0, rolling=0.0):
        pybullet.changeDynamics(
            self.objects[0],
            0,
            lateralFriction=lateral,
            spinningFriction=spinning,
            rollingFriction=rolling,
        )
