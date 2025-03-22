import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 5.0  # Mass
c = 0.1  # Damping coefficient
k = 1.0  # Spring stiffness
dt = 0.1  # Time step
T = 5.0  # Time horizon
N = int(T / dt)  # Number of steps

# Define symbolic variables (SX)
x1 = cs.SX.sym('x1')  # Position
x2 = cs.SX.sym('x2')  # Velocity
u = cs.SX.sym('u')    # Control input

# Define system dynamics
x1_dot = x2
x2_dot = (u - c * x2 - k * x1) / m
f = cs.Function('f', [x1, x2, u], [x1_dot, x2_dot])

# Define cost function (minimize control effort)
cost = u**2

# Define constraints (-1 <= u <= 1)
g = [u]
lbg = [-1]
ubg = [1]

# Formulate NLP problem
nlp = {
    'x': cs.vertcat(x1, x2, u),  # Decision variables
    'f': cost,  # Cost function
    'g': cs.vertcat(*g)  # Constraints
}

# Create solver
solver = cs.nlpsol('solver', 'ipopt', nlp)

# Initial state
x0 = [1, 1]  # Initial position and velocity

# Simulate the system
X = [x0]  # State trajectory
U = []    # Control trajectory
for _ in range(N):
    # Solve the optimization problem
    result = solver(x0=[X[-1][0], X[-1][1], 0], lbg=lbg, ubg=ubg)

    # Extract control input
    u_opt = float(result['x'][2])
    U.append(u_opt)

    # Simulate the system for one time step
    x_dot = f(X[-1][0], X[-1][1], u_opt)  # Evaluate dynamics
    x_dot_numeric = np.array([x_dot[0].full(), x_dot[1].full()]).flatten()  # Convert to NumPy array
    x_next = x_dot_numeric * dt + np.array(X[-1])  # Update state
    X.append(x_next.tolist())  # Convert to list for consistency

# Plot results
time = [i * dt for i in range(N + 1)]  # Time steps
x1_trajectory = [x[0] for x in X]  # Position trajectory
x2_trajectory = [x[1] for x in X]  # Velocity trajectory

# Plot position (x1) and velocity (x2)
plt.figure(figsize=(10, 6))
plt.plot(time, x1_trajectory, label='Position (x1)', color='blue')
plt.plot(time, x2_trajectory, label='Velocity (x2)', color='red')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.title('State vs Time')
plt.legend()
plt.grid()
plt.show()

# Plot control input (u)
plt.figure(figsize=(10, 6))
plt.plot(time[:-1], U, label='Control Input (u)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Control Input vs Time')
plt.legend()
plt.grid()
plt.show()