import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node

plot_on = True


def calculate_params(torques, q, qv, qa, threshold):
    indexes = np.where(abs(qv) >= -1000000)[0]
    q = q[indexes[20:-20]]
    qv = qv[indexes[20:-20]]
    qa = qa[indexes[20:-20]]
    ones = np.ones_like(q)
    torques = torques[indexes[20:-20]]
    Q = np.vstack([qa, np.cos(q), np.sin(q), qv,
                  np.arctan(qv/threshold)*2/np.pi]).T
    lam = 5 * np.eye(5, 5)
    lam[4, 4] = 10000
    return np.linalg.pinv((Q.T @ Q) + lam) @ Q.T @ torques


def evaluate(torques, q, qv, qa, x, threshold):
    # friction_velocity = np.where(
    #     abs(qv) > threshold, qv, 0)
    new_torques = qa*x[0] + np.cos(q)*x[1] + \
        np.sin(q)*x[2] + qv*x[3] + np.arctan(qv/threshold)*2/np.pi*x[4]
    return new_torques


def average_sum_of_squares(arr):
    return np.mean(np.square(arr))


def new_torque_and_plot(t, torques, q, qv, qa, x, threshold, index, fig, axs):
    new_torques = evaluate(torques, q, qv, qa, x, threshold)
    fig, axs = plt.subplots(2, 1)
    fig.suptitle("q"+str(index), fontsize=20)
    fig.subplots_adjust(left=0.05, right=0.993, top=0.945,
                        bottom=0.045, wspace=0.06, hspace=0.114)
    axs[0].plot(t[20:-20], np.vstack(
        [torques[20:-20], new_torques[20:-20]]).T, linestyle='-', linewidth=3, label=['set tau', 'tau'])
    axs[0].legend(fontsize=15)
    axs[0].set_xlabel("t [s]", fontsize=20 - 5, va='bottom')
    axs[0].set_ylabel("tau [Nm]", fontsize=20 - 5)
    axs[0].tick_params(axis='both', which='major', labelsize=20 - 7)
    # axs[index][0].set_title("Torques")
    axs[0].grid(True)
    axs[1].plot(t[20:-20], (torques - new_torques)[20:-20],
                linestyle='-', linewidth=3, label='tau error')
    axs[1].grid(True)
    axs[1].legend(fontsize=15)
    axs[1].set_xlabel("t [s]", fontsize=20 - 5, va='bottom')
    axs[1].set_ylabel("tau [Nm]", fontsize=20 - 5)
    axs[1].tick_params(axis='both', which='major', labelsize=20 - 7)
    axs[1].set_title("Torque error {}".format(
        average_sum_of_squares(torques - new_torques)))


def identify_friction(log_file, logger, plot_on):
    log_data = np.loadtxt(log_file, dtype=np.float64, delimiter=',')
    joints_num = (log_data.shape[1] - 2) // 3
    fig = [None] * joints_num
    axs = [None] * joints_num
    end_row = 0
    x = np.ndarray((5, joints_num), dtype=np.float64)
    thresholds = np.ndarray((joints_num), dtype=np.float64)
    q_index = 2
    for i in range(3):
        start_row = end_row
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
        qv = log_data[start_row:end_row, start_col:end_col]
        n_points = 31
        x_vals = np.arange(n_points)
        sigma = 12.0
        x_position = 15
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel/np.sum(kernel)
        plt.plot(kernel)
        plt.show()
        dt = (t[1000]-t[0])/1000
        derivative_kernel = np.array(
            [1.0/60, -3/20.0, 3/4.0, 0.0, -3/4.0, 3/20.0, -1/60.0])/dt

        index = q_index
        while index < joints_num:
            qv_filtered = np.convolve(qv[:, index], kernel, mode='same')
            qa_filtered = np.convolve(
                qv_filtered, derivative_kernel, mode='same')
            qa_filtered = np.convolve(
                qa_filtered, kernel, mode='same')
            minimum = 1000000
            for threshold in [0.05]:
                iks = calculate_params(
                    torques[:, index], q[:, index], qv_filtered, qa_filtered, threshold)
                error = average_sum_of_squares(
                    torques[:, index] - evaluate(torques[:, index], q[:, index], qv_filtered, qa_filtered, iks, threshold))
                if error < minimum:
                    minimum = error
                    thresholds[index] = threshold
                    x[:, index] = iks
            if plot_on:
                new_torque_and_plot(
                    t, torques[:, index], q[:, index], qv_filtered, qa_filtered, x[:, index], thresholds[index], index, fig, axs)

            index += 3
        q_index -= 1

    logger.info("B: {}".format(repr(x[3, :])))
    logger.info("Fv: {}".format(repr(x[4, :])))
    logger.info("sigma: {}".format(repr(thresholds)))
    if plot_on:
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = Node("identify_friction_node")
    node.declare_parameter("log_file", rclpy.Parameter.Type.STRING)
    node.declare_parameter("plot_on", True)
    log_file = node.get_parameter(
        "log_file").get_parameter_value().string_value
    plot_on = node.get_parameter(
        "plot_on").get_parameter_value().bool_value
    identify_friction(log_file, node.get_logger(), plot_on)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
