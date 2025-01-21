import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Define waypoints for upward and downward paths
T_up = 0.7  # Duration of upward path
T_down = 0.5  # Duration of downward path
T_total = T_up + T_down

y_points = 0.1469

# Upward path waypoints with a midpoint
t_up = np.array([0, T_up / 2, T_up])
x_up = np.array([0, 0.1, 0.2])  # X position with midpoint
z_up = np.array([0, 0.05, 0.1])  # Z height (peak) with midpoint

# Downward path waypoints with a midpoint
t_down = np.array([T_up, T_up + T_down / 2, T_total])
x_down = np.array([0.2, 0.1, 0])  # Return to start in X with midpoint
z_down = np.array([0.1, 0.05, 0])  # Return to ground in Z with midpoint


# Create splines for both paths
x_spline_up = CubicSpline(t_up, x_up)
z_spline_up = CubicSpline(t_up, z_up)

x_spline_down = CubicSpline(t_down, x_down)
z_spline_down = CubicSpline(t_down, z_down)

def time_scaling(t, T):
    """Fifth-order polynomial time scaling"""
    s_t = 10 * (t / T)**3 - 15 * (t / T)**4 + 6 * (t / T)**5
    s_dot = (30 * (t / T)**2 - 60 * (t / T)**3 + 30 * (t / T)**4) / T
    s_ddot = (60 * (t / T) - 180 * (t / T)**2 + 120 * (t / T)**3) / (T**2)
    return s_t, s_dot, s_ddot

# Evaluate trajectory
def evaluate_trajectory(t):
    t_mod = t % T_total  # Loop within [0, T_total]

    if t_mod <= T_up:  # Upward path
        s_t, s_dot, s_ddot = time_scaling(t_mod, T_up)
        x = x_spline_up(s_t)
        z = z_spline_up(s_t)
        x_vel = x_spline_up(s_t, 1) * s_dot
        z_vel = z_spline_up(s_t, 1) * s_dot
        x_acc = x_spline_up(s_t, 2) * (s_dot**2) + x_spline_up(s_t, 1) * s_ddot
        z_acc = z_spline_up(s_t, 2) * (s_dot**2) + z_spline_up(s_t, 1) * s_ddot
    else:  # Downward path
        t_down_mod = t_mod - T_up
        s_t, s_dot, s_ddot = time_scaling(t_down_mod, T_down)
        x = x_spline_down(s_t)
        z = z_spline_down(s_t)
        x_vel = x_spline_down(s_t, 1) * s_dot
        z_vel = z_spline_down(s_t, 1) * s_dot
        x_acc = x_spline_down(s_t, 2) * (s_dot**2) + x_spline_down(s_t, 1) * s_ddot
        z_acc = z_spline_down(s_t, 2) * (s_dot**2) + z_spline_down(s_t, 1) * s_ddot

    print(f"Current x position: {x:.5f}")

    return x, z, x_vel, z_vel, x_acc, z_acc

# Real-time simulation example
start_time = 0  # For demonstration
# Initialize lists to store data
time_data = []
x_data = []
z_data = []
x_vel_data = []
z_vel_data = []
x_acc_data = []
z_acc_data = []

for t in np.linspace(0, 2 * T_total, 500):
    # Evaluate trajectory at current time
    x, z, x_vel, z_vel, x_acc, z_acc = evaluate_trajectory(t)

    # Store data
    time_data.append(t)
    x_data.append(x)
    z_data.append(z)
    x_vel_data.append(x_vel)
    z_vel_data.append(z_vel)
    x_acc_data.append(x_acc)
    z_acc_data.append(z_acc)

# Plot position trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_data, z_data, label="Trajectory Path")
plt.scatter(x_up.tolist() + x_down.tolist(), z_up.tolist() + z_down.tolist(), color='red', label="Waypoints")
plt.xlabel("X Position")
plt.ylabel("Z Position")
plt.title("Trajectory Path with Time Scaling")
plt.legend()
plt.grid()
plt.show()

# Create subplots for position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Plot position
axs[0].plot(time_data, x_data, label='X Position')
axs[0].plot(time_data, z_data, label='Z Position')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].grid(True)

# Plot velocity
axs[1].plot(time_data, x_vel_data, label='X Velocity')
axs[1].plot(time_data, z_vel_data, label='Z Velocity')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].legend()
axs[1].grid(True)

# Plot acceleration
axs[2].plot(time_data, x_acc_data, label='X Acceleration')
axs[2].plot(time_data, z_acc_data, label='Z Acceleration')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Acceleration (m/s²)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
