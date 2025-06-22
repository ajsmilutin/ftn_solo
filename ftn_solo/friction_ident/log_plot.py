import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from scipy.signal import savgol_filter

log_data = np.loadtxt("ident_log.csv", dtype=np.float64, delimiter=',')
joints_num = (log_data.shape[1] - 2) // 3
# joints_num = 6
titles = ['Move knee', 'Move hip', 'Rotate hip']
fontsize = 20


def main(args=None):
    end_row = 0
    fig = [None] * joints_num
    axs = [None] * joints_num
    for i in range(joints_num):
        fig[i], axs[i] = plt.subplots(2, 2)
    q_index = 2
    for i in range(3):
        start_row = end_row + 1000
        # substract 1 from i for task joint spline
        end_row = np.where(log_data[:, 0] == float(2*i+1))[0][-1] + 1
        t = log_data[start_row:end_row, 1]
        start_col = 2
        end_col = 2 + joints_num
        torques = log_data[start_row:end_row, start_col:end_col]
        start_col = end_col
        end_col += joints_num
        q = log_data[start_row:end_row, start_col:end_col]
        start_col = end_col
        end_col += joints_num
        # qv = (q[2:] - q[:-2]) / (t[2:]-t[:-2])[:, np.newaxis]
        # qv = np.vstack([[0.0] * joints_num, qv, qv[-1]])
        # qa = (qv[2:] - qv[:-2]) / (t[2:]-t[:-2])[:, np.newaxis]
        # qa = np.vstack([[0.0] * joints_num, qa, qa[-1]])

        qv = log_data[start_row:end_row, start_col:end_col]

        n_points = 21
        x_vals = np.arange(n_points)
        sigma = 10.0
        x_position = 10
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel/np.sum(kernel)

        # qa1 = (qv1[1:]-qv1[:-1]) / (t[1:]-t[:-1])[:, np.newaxis]
        # qa1 = np.vstack([[0.0] * joints_num, qa1])
        qa = (qv[2:] - qv[:-2]) / (t[2:]-t[:-2])[:, np.newaxis]
        qa = np.vstack([[0.0] * joints_num, qa, qa[-1]])
        dt = t[1] - t[0]
        print("DT = {}".format(dt))
        derivative_kernel = np.array(
            [1.0/60, -3/20.0, 3/4.0, 0.0, -3/4.0, 3/20.0, -1/60.0])/dt
        # derivative_kernel = np.array(
        #     [1.0/2, -1.0/2])/dt            

        index = q_index
        while index < joints_num:
            fig[index].suptitle("q"+str(index % 3), fontsize=fontsize)
            fig[index].subplots_adjust(
                left=0.05, right=0.993, top=0.945, bottom=0.062, hspace=0.2, wspace=0.08)
            axs[index][0, 0].plot(t, torques[:, index],
                                  linestyle='-', linewidth=3)
            # axs[index][0].set_title("Torque", fontsize = fontsize - 5)
            axs[index][0, 0].grid(True)
            axs[index][0, 0].set_xlabel(
                "t [s]", fontsize=fontsize - 5, va='bottom')
            axs[index][0, 0].set_ylabel("tau [Nm]", fontsize=fontsize - 5)
            axs[index][0, 0].tick_params(
                axis='both', which='major', labelsize=fontsize - 7)
            axs[index][1, 0].plot(t, q[:, index], linestyle='-', linewidth=3)
            # axs[index][1].set_title("Position", fontsize = fontsize - 5)
            axs[index][1, 0].grid(True)
            axs[index][1, 0].set_xlabel(
                "t [s]", fontsize=fontsize - 5, va='bottom')
            axs[index][1, 0].set_ylabel("q [rad]", fontsize=fontsize - 5)
            axs[index][1, 0].tick_params(
                axis='both', which='major', labelsize=fontsize - 7)
            axs[index][0, 1].plot(t, qv[:, index], linestyle='-', linewidth=3)
            qv_filtered = np.convolve(qv[:, index], kernel, mode='same')
            qv_derived = np.convolve(q[:, index], derivative_kernel, mode='same')
            axs[index][0, 1].plot(t, qv_filtered, linestyle='-', linewidth=3)
            axs[index][0, 1].plot(t, qv_derived, linestyle='-', linewidth=3)
            
            

            axs[index][0, 1].grid(True)
            axs[index][0, 1].set_xlabel(
                "t [s]", fontsize=fontsize - 5, va='bottom')
            axs[index][0, 1].set_ylabel("qv [rad/s]", fontsize=fontsize - 5)
            axs[index][0, 1].tick_params(
                axis='both', which='major', labelsize=fontsize - 7)
            axs[index][1, 1].plot(t, qa[:, index], linestyle='-')
            axs[index][1, 1].set_title("Acceleration")
            axs[index][1, 1].grid(True)
            axs[index][1, 1].plot(t, qa[:, index], linestyle='-', linewidth=3)
            qa_filtered = np.convolve(
                qv_filtered, derivative_kernel, mode='same')
            axs[index][1, 1].plot(t, qa_filtered, linestyle='-', linewidth=3)
            #
            # axs[index][3].plot(t, qa_f[:, index], linestyle='--', linewidth=3)
            # axs[index][3].grid(True)
            # axs[index][3].set_xlabel("t [s]", fontsize=fontsize - 5, va = 'bottom')
            # axs[index][3].set_ylabel("qa [rad/s^2]", fontsize=fontsize - 5)
            # axs[index][3].tick_params(axis='both', which='major', labelsize=fontsize - 7)
            index += 3

        q_index -= 1

 #   plt.figure(2)
 #   plt.title('Move knee all torques')
 #   plt.plot(t, torques, linestyle='-')

 #   plt.figure(4)
 #   plt.title('Move hip all torques')
 #   plt.plot(t, torques, linestyle='-')

 #   plt.figure(6)
 #   plt.title('Rotate hip all torques')
 #   plt.plot(t, torques, linestyle='-')

    # plt.grid(True)
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
