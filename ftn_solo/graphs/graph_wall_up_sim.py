import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


cones = np.array([[-0.36, 0.14],
                  [-0.36, -0.14],
                  [0.36, 0.14],
                  [0.36, -0.14]])
plt.plot(cones[:, 0], cones[:, 1], color="white",  marker='o', markerfacecolor='None',
         markeredgecolor='black', markeredgewidth=3.0, markersize=18, label="Contact points")

fcontact = np.array([[9.99991662702753, 0.14770945434498614],
                     [9.999916579939132, 9.999883358220815],
                     [-10.00008334915391, 9.999883340971685],
                     [-10.000083298209045, 0.13641538192035788],
                     [-10.000083314436067, -10.000116129875073],
                     [9.999916600736846, -10.000116080023652],
                     [9.99991662702753, 0.14770945434498614]
                     ])
plt.plot(fcontact[:, 0], fcontact[:, 1],
         linewidth=3.0)

wall = np.array(
    [[0.36, 10], [0.36, -10], [-0.36, -10], [-0.36, 10], [0.36, 10]])
from numpy import genfromtxt


sample_data = genfromtxt("wall/com.csv", delimiter=",")
print(sample_data)    
plt.plot(sample_data[:,0], sample_data[:,1], 'r', linewidth=2.0, label="CoM")

sample_data = genfromtxt("wall/boundary.csv", delimiter=",")
plt.plot(sample_data[:,0], sample_data[:,1], 'k--', linewidth=2.0, label="First phase")
plt.plot(sample_data[:,6], sample_data[:,7], 'm-.', linewidth=2.0, label="Second phase")
plt.plot(sample_data[:,18], sample_data[:,19], 'b-', linewidth=3.0, label="Middle phase")
plt.plot(sample_data[:,27], sample_data[:,28], 'g--', linewidth=3.0, label="End fo motion")

des_com = genfromtxt("wall/des_com.csv", delimiter=",")
plt.plot(des_com[[0,2,5,9], 0], des_com[[0,2,5,9], 1],  marker='*', color=None, markerfacecolor='red',
          markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="Desired CoM")
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1], "k", label="wall",    linewidth=2.0)
ax = plt.gca()
plt.legend(loc="upper left")
plt.axis("equal")
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([-0.2, 0.2])
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("fig_wall_up_feasable.png", dpi=150)
plt.show()
