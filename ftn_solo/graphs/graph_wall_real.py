import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


wall = np.array(
    [[0.32, 10], [0.32, -10]])
from numpy import genfromtxt


com = genfromtxt("wall_real/com.csv", delimiter=",")
plt.plot(com[:,0], com[:,1], 'r', linewidth=2.0, label="CoM")

sample_data = genfromtxt("wall_real/boundary.csv", delimiter=",")
print(sample_data.shape)
plt.plot(sample_data[:,0], sample_data[:,1], 'k--', linewidth=2.0, label="First phase")
plt.plot(sample_data[:,2], sample_data[:,3], 'm-.', linewidth=2.0, label="Second phase")
plt.plot(sample_data[:,4], sample_data[:,5], 'b-', linewidth=3.0, label="Third phase")
plt.plot(sample_data[:,6], sample_data[:,7], 'g--', linewidth=3.0, label="Fourth phase")
plt.plot(sample_data[:,8], sample_data[:,9], 'r--', linewidth=3.0, label="End fo motion")
    
des_com = genfromtxt("wall_real/des_com.csv", delimiter=",")
plt.plot(des_com[:, 0], des_com[:, 1], linewidth=1.0, marker='*', color='r', markerfacecolor='red',
          markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="Desired CoM")
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1], "k", label="Wall",    linewidth=2.0)
ax = plt.gca()
plt.legend(loc="upper right")
plt.axis("equal")
ax.set_xlim([-0.25, 0.35])
ax.set_xlabel("X [m]")
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel("Y [m]")
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("fig_wall_real.png", dpi=150)
plt.show()
