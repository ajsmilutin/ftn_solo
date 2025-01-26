import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
import time


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class RneAlgorithm(PinocchioWrapper):
    def __init__(self, num_joints, yaml_config, robot_version, logger, dt) -> None:
        super().__init__(robot_version, logger, dt)
        # self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        # self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        self.B = float_or_list(yaml_config["B"], num_joints)
        self.Fv = float_or_list(yaml_config["Fv"], num_joints)
        # self.max_control = float_or_list(
        #     yaml_config["max_control"], num_joints)
        self.logger = logger
        self.q = np.zeros(19)
        self.dq_curr = np.array([])
        self.dq = np.array([])
        self.ndq = np.zeros(18)
        self.J_real =np.zeros((3, 18))
        self.J_dot =np.zeros((3, 18))
        self.q_base = np.array([0,0,0,0,0,0,1])
        self.dq_base = np.array([0,0,0,0,0,0])
        self.tau_con = np.zeros(12)
       

    def rnea(self, steps, acc, qcurr, dqcurr,qbase,t,kp):

        # qbase = np.array([qbase[1],qbase[2],qbase[3],qbase[0]])
        # self.q = np.concatenate((np.concatenate((self.q_base,qbase)), qcurr))
        self.q = np.concatenate((self.q_base,qcurr))
        self.dq_curr = np.concatenate((self.dq_base,dqcurr))
        self.ndq.fill(0)
        self.J_real.fill(0)
        self.J_dot.fill(0)
        new = self.framesForwardKinematics(self.q, self.dq_curr, self.end_eff_ids, steps)
        self.computeFrameJacobian(self.q,self.dq_curr)
        self.computeNonLinear(self.q, self.dq_curr)

        for x,end_eff_id in enumerate(self.end_eff_ids):
            J_real,J_dot = self.get_frame_jacobian(end_eff_id)
            self.dq = np.dot(np.linalg.pinv(J_real), new[x][:3])
            # dddq = np.dot(J_real,acc[x])
            # self.logger.info("ddq: {}".format(dddq))
            self.ndq += self.dq
            self.J_real += J_real
            self.J_dot += J_dot
        
        # self.logger.info(format(np.linalg.matrix_rank(J_real)))
        q = self.pinIntegrate(self.q, self.ndq)
      
        q_joints = q[7:19]
       
        ddq = self.pd_controller(q_joints, self.dq[6:], qcurr, dqcurr,t,kp)

        
      

        self.tau = self.compute_recrusive_newtone_euler(self.ndq, ddq,self.Fv,self.B,self.J_real[:3,6:],self.J_dot[:3,6:])
        return self.tau 
 