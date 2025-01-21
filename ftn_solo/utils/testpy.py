import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

# Define waypoints for trajectory
T = 2  # Total duration of trajectory (can be adjusted)\
y=0.1469
# Define waypoints for a smooth arc-like trajectory
#  np.array([0.196, 0.1469, -0.20]), -0.15Z 0.05X
t_arc_points = np.array([0,1.5,2])
x_arc_points = np.array([0.126, 0.196, 0.246])  # X positions
z_arc_points = np.array([-0.20, -0.15, -0.20])  # Z heights (arc)

t_line_points = np.array([1,1.5,2])
x_line_points = np.array([0.246, 0.196, 0.126])  # X positions
z_line_points = np.array([-0.20, -0.20, -0.20])  # Z heights (arc)

x_arc = CubicSpline(t_arc_points, x_arc_points)
z_arc = CubicSpline(t_arc_points, z_arc_points)
x_line = CubicSpline(t_line_points, x_line_points)
z_line = CubicSpline(t_line_points, z_line_points)

# Plot the splines for visualization
t_fine_up = np.linspace(0, 1, 5)
t_fine_down = np.linspace(1, 2, 100)

plt.figure(figsize=(10, 6))
plt.plot(x_arc(t_fine_up), z_arc(t_fine_up), label="Upward Path", color="blue")
plt.plot(x_arc(t_fine_up), label="X Path", color="yellow")
plt.plot(x_line(t_fine_down), z_line(t_fine_down), label="Downward Path", color="green")
plt.scatter(x_arc_points.tolist() + z_arc_points.tolist(), x_line_points.tolist() + z_line_points.tolist(), color='red', label="Waypoints")
plt.xlabel("X Position")
plt.ylabel("Z Position")
plt.title("Spline Visualization: Upward and Downward Paths")
plt.legend()
plt.grid()
plt.show()




# Function to evaluate trajectory at any given time t

def get_trajectory(t,T):
    steps= []
    T_total = T + 2
    # Normalize time within [0, T] using modulo
    t_mod = t % T_total

    # Cubic time scaling    
    # Fifth-order polynomial time scaling (quintic time scaling)
    if t_mod <= T:
        s_t = 10 * (t_mod / T)**3 - 15 * (t_mod / T)**4 + 6 * (t_mod / T)**5
        x = x_arc(s_t)
        z = z_arc(s_t)
    else:
        t_d = t_mod - T
        s_t = 10 * (t_d / T)**3 - 15 * (t_d / T)**4 + 6 * (t_d / T)**5
        x = x_line(s_t)
        z = z_line(s_t)
            

    return x, z

# Real-time simulation example
start_time = time.time()
# Initialize lists to store data
time_data = []
x_data = []
z_data = []
x_vel_data = []
z_vel_data = []
x_acc_data = []
z_acc_data = []

while True:
    # Get current simulation time
    t = time.time() - start_time
    leg_position = []

    # Evaluate trajectory at current time
    x, z= get_trajectory(t,T)
    print(x,z)  
   


    # leg_position.append([x,y,z])
   
    # print(leg_position)
    
    # Exit loop after 10 seconds for demonstration
    if t > 4:
        break

    # Delay to simulate real-time processing (50 ms per step)
    time.sleep(0.05)

