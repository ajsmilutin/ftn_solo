from shlex import join
from matplotlib import figure
import numpy as np
import matplotlib.pyplot as plt

log_data = np.loadtxt("test_comp.log", dtype=np.float64, delimiter=',')
joints_num = (log_data.shape[1] - 2) // 3
# joints_num = 6
# titles = ['Move knee', 'Move hip', 'Rotate hip']


def main(args=None):
    end_row = 0
    fig = [None] * joints_num
    axs = [None] * joints_num
    for i in range(joints_num):
        fig[i], axs[i] = plt.subplots(2, 2)
    t = log_data[:, 1]
    start_col = 2
    end_col = 2 + joints_num
    torques = log_data[:, start_col:end_col]
    start_col = end_col
    end_col += joints_num
    q = log_data[:, start_col:end_col]
    start_col = end_col
    end_col += joints_num
    qv = log_data[:, start_col:end_col]
    qa = (qv[2:] - qv[:-2]) / (t[2:]-t[:-2])[:, np.newaxis]
    qa = np.vstack([[0.0] * joints_num, qa, qa[-1]])
    index = 0
    # fig[i-1], axs[i-1] = plt.subplots(2, 2)
    while index < joints_num:
        fig[index].suptitle("q"+str(index))
        axs[index][0, 0].plot(t, torques[:, index], linestyle='-')
        axs[index][0, 0].set_title("Torque")
        axs[index][0, 1].plot(t, q[:, index], linestyle='-')
        axs[index][0, 1].set_title("Position")
        axs[index][1, 0].plot(t, qv[:, index], linestyle='-')
        axs[index][1, 0].set_title("Velocity")
        axs[index][1, 1].plot(t, qa[:, index], linestyle='-')
        axs[index][1, 1].set_title("Acceleration")
        # axs[i-1][ind, 0].plot(t, qv[:, index], linestyle='-')
        # axs[i-1][ind, 1].plot(t, qa[:, index], linestyle='-')
        # axs[i-1][ind, 2].plot(t, qv1[:, index], linestyle='-')
        # axs[i-1][ind, 3].plot(t, qa1[:, index], linestyle='-')
        index += 1


 #   plt.figure(2)
 #   plt.title('Move knee all torques')
 #   plt.plot(t, torques, linestyle='-')

 #   plt.figure(4)
 #   plt.title('Move hip all torques')
 #   plt.plot(t, torques, linestyle='-')

 #   plt.figure(6)
 #   plt.title('Rotate hip all torques')
 #   plt.plot(t, torques, linestyle='-')

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
