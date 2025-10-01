import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


wall = np.array([[0.36, 10], [0.36, -10], [-0.36, -10], [-0.36, 10], [0.36, 10]])
from numpy import genfromtxt


com = genfromtxt("climb/com.csv", delimiter=",")
plt.plot(com[:, 0], com[:, 1], "r", linewidth=2.0, label="CoM")

sample_data = genfromtxt("climb/boundary.csv", delimiter=",")
# print(sample_data.shape)
plt.plot(
    sample_data[:, 0], sample_data[:, 1], "k--", linewidth=2.0, label="First phase"
)
plt.plot(
    sample_data[:, 2], sample_data[:, 3], "m-.", linewidth=2.0, label="Second phase"
)
plt.plot(sample_data[:, 4], sample_data[:, 5], "b-", linewidth=3.0, label="Third phase")
plt.plot(
    sample_data[:, 6], sample_data[:, 7], "g--", linewidth=3.0, label="Fourth phase"
)

des_com = genfromtxt("climb/des_com.csv", delimiter=",")
plt.plot(
    des_com[:, 0],
    des_com[:, 1],
    linewidth=1.0,
    marker="*",
    color="r",
    markerfacecolor="red",
    markeredgecolor="red",
    markeredgewidth=1.0,
    markersize=10,
    label="Desired CoM",
)
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1], "k", label="Wall", linewidth=2.0)
cones = np.array([[-0.36, 0.12],
                  [-0.36, -0.12],
                  [0.36, 0.12],
                  [0.36, -0.12]])
plt.plot(cones[:, 0], cones[:, 1], color="none",  marker='o', markerfacecolor='None',
         markeredgecolor='black', markeredgewidth=3.0, markersize=12, label="Contact points")

cones = np.array([[-0.15, 0.12],
                  [-0.15, -0.12],
                  [0.15, 0.12],
                  [0.15, -0.12]])
plt.plot(cones[:, 0], cones[:, 1], color="none",  marker='o', markerfacecolor='None',
         markeredgecolor='red', markeredgewidth=3.0, markersize=12, label="Initial feet pos.")         

ax = plt.gca()
plt.legend(loc="lower left")
plt.axis("equal")
ax.set_xlim([-0.4, 0.4])
ax.set_xlabel("X [m]")
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel("Y [m]")
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("fig_climb_real.png", dpi=150)
plt.show()


torque = genfromtxt("climb/efforts.csv", delimiter=",")
joints= [2,3,5,7,8,9, 11,12]
plt.plot(torque[:, 0]-torque[0, 0], torque[:, joints],linewidth=2.0, label=["FL Hip", "FL Knee", "FR Hip", "FR Knee", "BL Hip", "BL Knee", "BR Hip", "BR Knee"])
plt.legend(loc="lower left")
A = np.array([[-10, 2.5], [50, 2.5], [50, -2.5], [-10, -2.5]])
plt.plot(A[:, 0], A[:,1], linewidth=2, color='k')

ax = plt.gca()
ax.set_ylim([-2.7, 2.7])
ax.set_ylabel("tau [Nm]")
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
ax.set_xlim([0, 45])
ax.set_xlabel("t [s]")
plt.show()