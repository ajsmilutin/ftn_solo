import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper


def float_or_list(value, num_joints):
    return np.array(value if type(value) is list else [float(value)]*num_joints, dtype=np.float64)


class RneAlgorithm(PinocchioWrapper):
    def __init__(self, num_joints, yaml_config, robot_version, logger, dt) -> None:
        super().__init__(robot_version, logger, dt)
        # self.Kp = float_or_list(yaml_config["Kp"], num_joints)
        # self.Kd = float_or_list(yaml_config["Kd"], num_joints)
        # self.B = float_or_list(yaml_config["B"], num_joints)
        # self.max_control = float_or_list(
        #     yaml_config["max_control"], num_joints)
        self.q = np.zeros(19)

    def rnea(self, steps, qcurr, dqcurr):

        for x, joints in enumerate(self.end_eff_ids):
            n = self.framesForwardKinematics(
                self.q, joints, steps[x], self.base_link, x)
            J = self.computeFrameJacobian(self.q, joints)
            dq = np.dot(J, n)
            q = self.pinIntegrate(self.q, dq)
            q_joints = q[7:19]
            ref_tau = self.pd_controller(q_joints, dq[6:18], qcurr, dqcurr)
            ddq = self.get_acceleration(q, dq, ref_tau)
            self.tau = self.compute_recrusive_newtone_euler(q, dq, ddq)
            self.q = q

        return self.tau[6:18]
