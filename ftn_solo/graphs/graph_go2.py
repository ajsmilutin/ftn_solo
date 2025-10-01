import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


cones = np.array([[-0.25, 0.14],
                  [-0.25, -0.14],
                  [0.45, 0.14],
                  [0.45, -0.14]])
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
    [[0.45, 10], [0.45, -10]])
from numpy import genfromtxt


sample_data = genfromtxt("go2/com.csv", delimiter=",")
print(sample_data)    
plt.plot(sample_data[:,0], sample_data[:,1], 'r', linewidth=2.0, label="CoM")

sample_data = genfromtxt("go2/pera.csv", delimiter=",")
sample_data = np.column_stack((sample_data, sample_data[:, 0]))
plt.plot(sample_data[:,0], sample_data[:,1], 'k--', linewidth=2.0, label="First phase")
plt.plot(sample_data[:,2], sample_data[:,3], 'm-.', linewidth=2.0, label="Second phase")
plt.plot(sample_data[:,11], sample_data[:,11], 'b-', linewidth=3.0, label="Middle phase")
plt.plot(sample_data[:,16], sample_data[:,17], 'g--', linewidth=3.0, label="End fo motion")

des_com = genfromtxt("go2/des_com.csv", delimiter=",")
plt.plot(des_com[[0,2,5,8], 0], des_com[[0,2,5,8], 1],  marker='*', color=None, markerfacecolor='red',
          markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="Desired CoM")
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1], "k", label="wall",    linewidth=2.0)
ax = plt.gca()
plt.legend(loc="lower right")
plt.axis("equal")
ax.set_xlim([-0.4, 0.8])
ax.set_xlabel("X [m]")
ax.set_ylim([-0.3, 0.3])
ax.set_ylabel("Y [m]")  
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("go2_graph.png", dpi=150)
plt.show()
