import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


wall = np.array(
    [[0.2, 10], [0.2, -10], [0.39, -10], [0.39, 10], [0.39, 10]])
from numpy import genfromtxt


com = genfromtxt("slope/com.csv", delimiter=",")
plt.plot(com[:,0], com[:,1], 'r', linewidth=2.0, label="CoM")

sample_data = genfromtxt("slope/boundary.csv", delimiter=",")
print(sample_data.shape)
plt.plot(sample_data[:,0], sample_data[:,1], 'k--', linewidth=2.0, label="First phase")
plt.plot(sample_data[:,33], sample_data[:,34], 'm-.', linewidth=2.0, label="Front feet")
plt.plot(sample_data[:,60], sample_data[:,64], 'b-', linewidth=3.0, label="Back feet")
plt.plot(sample_data[:,102], sample_data[:,103], 'g--', linewidth=3.0, label="End fo motion")

des_com = genfromtxt("slope/des_com.csv", delimiter=",")
plt.plot(des_com[:, 0], des_com[:, 1], linewidth=1.0, marker='*', color='r', markerfacecolor='red',
          markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="Desired CoM")
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1], "k", label="Slope",    linewidth=2.0)
ax = plt.gca()
plt.legend(loc="upper left")
plt.axis("equal")
ax.set_xlim([-0.2, 0.7])
ax.set_xlabel("X [m]")
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel("Y [m]")
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("fig_slope_feasable.png", dpi=150)
plt.show()
