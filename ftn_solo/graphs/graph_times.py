import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from numpy import genfromtxt
from scipy import stats

sample_data = genfromtxt("times/cones_4.csv", delimiter=",")
sel = sample_data[:, 0]==3
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'r.', linewidth=2.0, label="4 sided 3 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'r-', label='4 sided 3 contacts fitted line')

sel = sample_data[:, 0]==4
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'r+', linewidth=2.0, markersize=4, label="4 sided 4 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'r-.', label='4 sided 4 contacts fitted line')

sample_data = genfromtxt("times/cones_6.csv", delimiter=",")
sel = sample_data[:, 0]==3
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'b.', linewidth=2.0, label="6 sided 3 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'b-', label='6 sided 3 contacts fitted line')

sel = sample_data[:, 0]==4
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'b+', linewidth=2.0, label="6 sided 4 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'b-.', label='6 sided 4 contacts fitted line')


sample_data = genfromtxt("times/cones.csv", delimiter=",")
sel = sample_data[:, 0]==3
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'g.', linewidth=2.0, label="8 sided, 3 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'g-', label='8 sided, 3 contacts')

sel = sample_data[:, 0]==4
plt.plot((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0), 'g+', linewidth=2.0, label="8 sided, 4 contacts")
res = stats.linregress((sample_data[sel,3]-1), (sample_data[sel,2]/1000.0))
plt.plot((sample_data[sel,3]-1), res.intercept + res.slope*(sample_data[sel,3]-1), 'g-.', label='8 sided, 4 contacts')


# sample_data = genfromtxt("wall/boundary.csv", delimiter=",")
# plt.plot(sample_data[:,0], sample_data[:,1], 'k--', linewidth=2.0, label="First phase")
# plt.plot(sample_data[:,6], sample_data[:,7], 'm-.', linewidth=2.0, label="Second phase")
# plt.plot(sample_data[:,18], sample_data[:,19], 'b-', linewidth=3.0, label="Middle phase")
# plt.plot(sample_data[:,27], sample_data[:,28], 'g--', linewidth=3.0, label="End fo motion")

# des_com = genfromtxt("wall/des_com.csv", delimiter=",")
# plt.plot(des_com[[0,2,5,9], 0], des_com[[0,2,5,9], 1],  marker='*', color=None, markerfacecolor='red',
#           markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="Desired CoM")
plt.grid(visible=True)
# plt.plot(wall[:, 0], wall[:, 1], "k", label="wall",    linewidth=2.0)
ax = plt.gca()
ax.set_xlabel("num edges")
ax.set_ylabel("computation time[ms]")
plt.legend(loc="upper left")
# plt.axis("equal")
# ax.set_xlim([-0.4, 0.4])
# ax.set_ylim([-0.2, 0.2])
fig = plt.gcf()
fig.set_size_inches(10.5, 7)
fig.savefig("times.png", dpi=150)
plt.show()
