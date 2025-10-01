import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


cones = np.array([[-0.36, -0.14],
                  [0.36, 0.14],
                  [0.36, -0.14]])
plt.plot(cones[:, 0], cones[:, 1], color="white",  marker='o', markerfacecolor='None',
         markeredgecolor='black', markeredgewidth=3.0, markersize=18, label="Contact points")

fcontact = np.array([
    [10.061386110281537, 3.7825790289503276],
    [10.061386106681665, 3.9043881206518622],
    [-9.938609707024543, -3.8565154044321766],
    [-9.938613905952222, -10.049294854738763],
    [10.061385936553277, -10.049294648098673],
    [10.061386110281537, 3.7825790289503276]
])
plt.plot(fcontact[:, 0], fcontact[:, 1],
         linewidth=3.0)

fa_in = np.array([[0.3525755313589545, -0.07600942223920662],
                  [0.3518721746784366, 0.03440265673716033],
                  [0.18626221539282112, 0.043235880375697616],
                  [0.10614539709479988, 0.04130431788898566],
                  [-0.08389508386416861, -0.03243998677518416],
                  [-0.17629294573622906, -0.1301496914845138],
                  [-0.17313763371732477, -0.1630306825329872],
                  [-0.16353376994825408, -0.17985739387803096],
                  [-0.12338206555786263, -0.22531587334259773],
                  [0.03943345878338144, -0.22390450352328797],
                  [0.1639513655705577, -0.22192861625338436],
                  [0.2757607601015977, -0.18465784656087672],
                  [0.3245267295791355, -0.12568871753738625],
                  [0.35153078918027864, -0.08152757925811832],
                  [0.3525755313589545, -0.07600942223920662],
                  ])


plt.plot(fa_in[:, 0], fa_in[:, 1], "m--", linewidth=3,         )

wall =np.array([[0.36, 10], [0.36, -10], [-0.36, -10], [-0.36, 10],[0.36, 10]])

plt.plot(0.04, 0,  marker='*', color="None", markerfacecolor='red',
         markeredgecolor='red', markeredgewidth=1.0, markersize=10, label="COM")
plt.grid(visible=True)
plt.plot(wall[:, 0], wall[:, 1],"k", label="wall",
         linewidth=3.0)
ax = plt.gca()
ax.add_patch(Polygon(fcontact, fill=False, hatch="/", linewidth=1.0, ec="b", label="Contact support area") )
ax.add_patch(Polygon(fa_in, fill=False, hatch="\\", linewidth=1.0, ec="m", label="Feasible area "))
plt.legend(loc="upper left")
plt.axis("equal")
ax.set_xlim([-0.4, 0.4])
ax.set_ylim([-0.2, 0.2])
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
fig.savefig("fig3.png", dpi=150)
plt.show()
