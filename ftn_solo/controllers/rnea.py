import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper


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
        self.q_base = np.array([0,0,0,0,0,0,1])
        self.dq_base = np.array([0,0,0,0,0,0])
        self.tau_con = np.zeros(12)

    def rnea(self, steps, qcurr, dqcurr,qbase):

        # qbase = np.array([qbase[1],qbase[2],qbase[3],qbase[0]])
        # self.q = np.concatenate((np.concatenate((self.q_base,qbase)), qcurr))
        self.q_curr = np.concatenate((self.q_base,qcurr))
        self.dq_curr = np.concatenate((self.dq_base,dqcurr))
        self.ndq.fill(0)
        self.tau_con.fill(0)
        new = self.framesForwardKinematics(
            self.q_curr, self.end_eff_ids, steps, self.base_link)
        self.computeFrameJacobian(self.q_curr,self.dq_curr)
        self.computeNonLinear(self.q_curr, self.dq_curr)

        # for x,end_eff_id in enumerate(self.end_eff_ids):
        J, J_real,J_dot = self.get_frame_jacobian(self.end_eff_ids[0])
        
        self.dq = np.dot(J, new[0]) 
        
        self.ndq += self.dq
        
        # self.logger.info("tau_con: {}".format(J.T[:, 6:]))
        q = self.pinIntegrate(self.q_curr, self.ndq)
      
       
        ddq = self.pd_controller(q[7:19], self.dq[6:18], qcurr, dqcurr)

        
        self.tau = self.compute_recrusive_newtone_euler(self.ndq, ddq,self.Fv,self.B,J.T)
        return self.tau 
 