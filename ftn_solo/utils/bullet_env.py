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
    """This class manages a PyBullet simulation environment and provides utility functions to interact with :py:obj:`PinBulletWrapper` objects.

    Attributes:
        dt (float): The length of the simulation integration step.
        objects (list): The list of the PyBullet ids for all the non-robot objects.
        robots (list): The list of the robot wrapper of all added robots.
    """

    def __init__(self,robot_version, server=pybullet.GUI, dt=0.001):
        """Initializes the PyBullet client.

        Args:
            server (int, optional): PyBullet server mode. pybullet.GUI creates a graphical frontend using OpenGL while pybullet.DIRECT does not. Defaults to pybullet.GUI.
            dt (float, optional): The length of the simulation integration step.. Defaults to 0.001.
        """
        self.dt = dt
        self.objects = []
        self.robots = []

        self.physics_client = pybullet.connect(server)
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setPhysicsEngineParameter(fixedTimeStep=dt, numSubSteps=1)
        self.resources = Resources(robot_version)

    def add_robot(self, robot):
        self.robots.append(robot)
        return robot

    def add_object_from_urdf(
        self, urdf_path, pos=[0, 0, 0], orn=[0, 0, 0, 1], useFixedBase=True
    ):
        # Load the object.
        object_id = pybullet.loadURDF(urdf_path, useFixedBase=useFixedBase)
        pybullet.resetBasePositionAndOrientation(object_id, pos, orn)
        self.objects.append(object_id)
        return object_id

    def start_video_recording(self, file_name):
     
        self.file_name = file_name
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_video_recording(self):
        """Stops video recording if any."""
        if hasattr(self, "file_name"):
            pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def step(self, sleep=False):
        """Integrates the simulation one step forward.

        Args:
            sleep (bool, optional): Determines if the simulation sleeps for :py:attr:`~dt` seconds at each step. Defaults to False.
        """
        if sleep:
            time.sleep(self.dt)
        pybullet.stepSimulation()

        

    def print_physics_engine_params(self):
        """Prints the parametes of the physics engine."""
        params = pybullet.getPhysicsEngineParameters(self.physicsClient)
        print("physics_engine_params:")
        for key in params:
            print("    - ", key, ": ", params[key])


class BulletEnvWithGround(BulletEnv):
    """This class provides a shortcut to construct a PyBullet simulation environment with a flat ground.
    parent:
    """

    def __init__(self,robot_version, server=pybullet.GUI, dt=0.001):
        super().__init__(robot_version,server, dt)
        # with importlib_resources.path(__package__, "bullet_env.py") as p:
        #     package_dir = p.parent.absolute()
        #plane_urdf = str(package_dir / "resources" / "plane_with_restitution.urdf")
        plane_urdf = self.resources.urdf_plane_path
        self.add_object_from_urdf(plane_urdf)

    """Sets friction coefficients of the env. floor
     Args:
        lateral (float, optional): The lateral friction coefficient of the env. floor
        spinning (float, optional): The spinning friction coefficient of the env. floor
        rolling (float, optional): The rolling friction coefficient of the env. floor
    """

    def set_floor_frictions(self, lateral=1.0, spinning=0.0, rolling=0.0):
        pybullet.changeDynamics(
            self.objects[0],
            0,
            lateralFriction=lateral,
            spinningFriction=spinning,
            rollingFriction=rolling,
        )
