import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R

def cycloid_trajectory(R, T, num_points):
    """
    Generate a cycloid trajectory for an end effector.

    Parameters:
    R (float): Radius of the cycloid (determines the height).
    T (float): Total duration of the movement.
    num_points (int): Number of points along the trajectory.

    Returns:
    np.ndarray: Array of shape (num_points, 3) representing the 3D trajectory.
    """
    t = np.linspace(0, T, num_points)
    
    x_o = R * (t - np.sin(t))
    z_o = R * (1 - np.cos(t))
    
    z=-0.2+z_o#Legit
    x=0.196+x_o
    

    y = np.zeros_like(t)  # Assuming movement in x-z plane


    d = np.sqrt(x_o**2+z**2)
    
    beta = np.arccos(d/0.32)

    teta = np.arccos(abs(z)/d)
    

    alfa = beta + teta

    
   
    angle = np.degrees(alfa)

    print(f"position: {x},{y},{z}")
    print(angle) 


    trajectory = np.vstack((x_o, y, z)).T
    return trajectory

# Parameters for the cycloid
R = 0.025  # Radius (height of the cycloid)
T =  np.pi  # Total duration (in radians)
num_points = 10  # Number of points in the trajectory

# Generate the trajectory
trajectory = cycloid_trajectory(R, T, num_points)

# Plot the trajectory
plt.plot(trajectory[:, 0], trajectory[:, 2])
plt.xlabel('X Position')
plt.ylabel('Z Position')
plt.title('Cycloid Trajectory for End Effector')
plt.grid()
plt.axis('equal')
plt.show()
