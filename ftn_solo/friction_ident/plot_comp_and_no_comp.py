from cProfile import label
from shlex import join
from matplotlib import figure
from matplotlib.widgets import EllipseSelector
import numpy as np
import matplotlib.pyplot as plt

log_comp = np.loadtxt("test_comp_4.csv", dtype=np.float64, delimiter=',')
log_no_comp = np.loadtxt("test_no_comp_4.csv", dtype=np.float64, delimiter=',')
joints_num = (log_comp.shape[1] - 2) // 3
joint_sin_pos = np.array([0.5, 1.0, 1.57], dtype=np.float64)
SIN_W = 2 * 3.14 * 0.5
# joints_num = 6
# titles = ['Move knee', 'Move hip', 'Rotate hip']

def extract_data(log_data, phase):
    if phase == 1.0:
        start_row = 0
    else:
        start_row = np.where(log_data[:, 0] == phase - 1)[0][-1] + 1
    end_row = np.where(log_data[:, 0] == phase)[0][-1] + 1
    start_col = 2
    end_col = 2 + joints_num
    t = log_data[start_row:end_row, 1]
    torques = log_data[start_row:end_row, start_col:end_col]
    start_col = end_col
    end_col += joints_num
    position = np.tile(joint_sin_pos, (joints_num // 3)).reshape(joints_num, 1)
    if phase == 1.0:
        factor = 1.0
    elif phase == 2.0:
        factor = 0.8
    else:
        factor = 0.4
    q_ref = np.sin(SIN_W * factor * (t - t[0])).reshape(end_row-start_row,1).dot(position.T)
    q = log_data[start_row:end_row, start_col:end_col]
    start_col = end_col
    end_col += joints_num
    qv = log_data[start_row:end_row, start_col:end_col]
    qa = (qv[2:] - qv[:-2]) / (t[2:]-t[:-2])[:, np.newaxis]
    qa = np.vstack([[0.0] * joints_num, qa, qa[-1]])
    return t, q_ref, torques, q, qv, qa


def main(args=None):
    fig = [None] * joints_num
    axs = [None] * joints_num
    for i in range(joints_num):
        fig[i], axs[i] = plt.subplots(3, 1)
    ident_phase = 1.0
    t_comp, q_ref_comp, torques_comp, q_comp, qv_comp, qa_comp = extract_data(log_comp, ident_phase)
    t_no_comp, q_ref_no_comp, torques_no_comp, q_no_comp, qv_no_comp, qa_no_comp = extract_data(log_no_comp, ident_phase)
    index = 0
    # fig[i-1], axs[i-1] = plt.subplots(2, 2)
    while index < joints_num:
        fig[index].suptitle("q"+str(index % 3),fontsize = 20)
        fig[index].subplots_adjust(left=0.05, right=0.993, top=0.945, bottom=0.045, hspace=0.2, wspace=0.08)
        axs[index][0].plot(t_no_comp, torques_no_comp[:, index], linestyle='-', linewidth=3, label='No comp')
        axs[index][0].plot(t_comp, torques_comp[:, index], linestyle='-', linewidth=3, label='Comp')
        #axs[index][0].set_title("Torque")
        axs[index][0].legend(fontsize=15)
        axs[index][0].grid(True)
        axs[index][0].set_xlabel("t [s]", fontsize=20 - 5, va = 'bottom')
        axs[index][0].set_ylabel("tau [Nm]", fontsize=20 - 5)
        axs[index][0].tick_params(axis='both', which='major', labelsize=20 - 7)  
        #axs[index][0, 1].plot(t_no_comp, q_ref_no_comp[:, index], linestyle='-', label='Ref No comp')
        axs[index][1].plot(t_comp, q_ref_comp[:, index], linestyle='-', linewidth=3, label='Ref', color='green')
        axs[index][1].plot(t_no_comp, q_no_comp[:, index], linestyle='-', linewidth=3, label='No comp')
        axs[index][1].plot(t_comp, q_comp[:, index], linestyle='-', linewidth=3, label='Comp')
        #axs[index][1].set_title("Position")
        axs[index][1].legend(fontsize=15)
        axs[index][1].grid(True)
        axs[index][1].set_xlabel("t [s]", fontsize=20 - 5, va = 'bottom')
        axs[index][1].set_ylabel("q [rad]", fontsize=20 - 5)
        axs[index][1].tick_params(axis='both', which='major', labelsize=20 - 7)  
        axs[index][2].plot(t_no_comp, qv_no_comp[:, index], linestyle='-', linewidth=3, label='No comp')
        axs[index][2].plot(t_comp, qv_comp[:, index], linestyle='-', linewidth=3, label='Comp')
        #axs[index][2].set_title("Velocity")        
        axs[index][2].legend(fontsize=15)
        axs[index][2].grid(True)
        axs[index][2].set_xlabel("t [s]", fontsize=20 - 5, va = 'bottom')
        axs[index][2].set_ylabel("qv [rad/s]", fontsize=20 - 5)
        axs[index][2].tick_params(axis='both', which='major', labelsize=20 - 7) 
        #axs[index][1, 1].plot(t_no_comp, qa_no_comp[:, index], linestyle='-', label='No comp')
        #axs[index][1, 1].plot(t_comp, qa_comp[:, index], linestyle='-', label='Comp')
        #axs[index][1, 1].set_title("Acceleration")
        #axs[index][1, 1].legend()
        #axs[index][1, 1].grid(True)
        #fig[index].savefig('/home/vladam/phase 3/q'+str(index), dpi=300)
        index += 1
        if (joints_num == 13) and (index == 9):
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
    #plt.tight_layout()
    #plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
