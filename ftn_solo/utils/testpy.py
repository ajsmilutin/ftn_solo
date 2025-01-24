import numpy as np

# Quintic polynomial time scaling
def poly5(t, t_start, t_end):
    delta_t = t_end - t_start
    tau = (t - t_start) / delta_t
    result = np.zeros(3)
    result[0] = (((6 * tau - 15) * tau) + 10) * tau * tau * tau
    result[1] = (((30 * tau - 60) * tau) + 30) * tau * tau / delta_t
    result[2] = (((120 * tau - 180) * tau) + 60) * tau / delta_t / delta_t
    return result

# Compute interpolation for linear trajectories
def compute_interpolation(s, p_start, p_end):
    direction = p_end - p_start
    pos = p_start + s[0] * direction
    vel = s[1] * direction
    acc = s[2] * direction
    return pos, vel, acc

# Compute interpolation for rotational trajectories
def compute_interpolation_rotation(s, q_start, q_end):
    rotation_vector = np.log(np.linalg.inv(q_start) @ q_end)
    pos = q_start @ np.exp(s[0] * rotation_vector)
    vel = s[1] * q_start @ rotation_vector
    acc = s[2] * q_start @ rotation_vector
    return pos, vel, acc

class PiecewiseLinearTrajectory:
    def __init__(self):
        self.points = []
        self.times = []
        self.loop = False

    def add_point(self, point, time):
        self.points.append(np.array(point))
        self.times.append(time)

    def close_loop(self, time):
        self.loop = True
        self.add_point(self.points[0], time)

    def evaluate(self, t):
        if self.loop:
            t = t % self.times[-1]

        segment = 0
        while segment < len(self.times) - 1 and t > self.times[segment + 1]:
            segment += 1

        t_start = self.times[segment]
        t_end = self.times[segment + 1]
        s = poly5(t, t_start, t_end)

        p_start = self.points[segment]
        p_end = self.points[segment + 1]

        return compute_interpolation(s, p_start, p_end)

    def zero_position(self):
        return np.zeros_like(self.points[0]) if self.points else np.zeros(3)

    def zero_velocity(self):
        return np.zeros_like(self.points[0]) if self.points else np.zeros(3)

# Example usage
trajectory = PiecewiseLinearTrajectory()
trajectory.add_point([0, 0, 0], 0)
trajectory.add_point([0.1, 0.05, 0.2], 1)
trajectory.add_point([0.2, 0.1, 0], 2)
trajectory.add_point([0, 0, 0], 3)
trajectory.close_loop(4)

time_data = np.linspace(0, 4, 500)
pos_data = []
vel_data = []
acc_data = []

for t in time_data:
    pos, vel, acc = trajectory.evaluate(t)
    pos_data.append(pos)
    vel_data.append(vel)
    acc_data.append(acc)

pos_data = np.array(pos_data)
vel_data = np.array(vel_data)
acc_data = np.array(acc_data)

# Print out results for debugging
print("Position Data:\n", pos_data)
print("Velocity Data:\n", vel_data)
print("Acceleration Data:\n", acc_data)
