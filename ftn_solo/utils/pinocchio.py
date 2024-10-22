import pinocchio as pin
import numpy as np
from robot_properties_solo.solo12wrapper import Solo12Config


class PinocchioWrapper(object):
    def __init__(self, robot_version, logger, dt):

        # self.resourses = Resources(robot_version)
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.model = self.pin_robot.model
        self.data = self.model.createData()
        self.logger = logger
        self.dt = dt
        self.prev_err = np.inf
        self.delta_error = np.inf
        self.nu = []

        self.end_eff_ids = []
        self.end_effector_names = []
        controlled_joints = []

        for leg in ["FL", "FR", "HL", "HR"]:
            controlled_joints += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
            self.end_eff_ids.append(
                self.model.getFrameId(leg + "_ANKLE")
            )
            self.end_effector_names.append(leg + "_ANKLE")

        self.base_link = self.model.getFrameId("base_link")

        self.fr = pin.ReferenceFrame.LOCAL

        # Parameters for IK
        self.DT = 0.01
        self.epsilon = 0.00000001
        self.k_max = 0.00000002
        self.J = np.zeros((6, 18))
        self.J_list=[]

    def mass(self, q):
        return pin.crba(self.model, self.data, q)

    def gravity(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    def moveSE3(self, R, T):
        oMdes = pin.SE3(R, T)
        return oMdes

    def pinIntegrate(self, q, v):
        return pin.integrate(self.model, q, v*self.DT)

    def calculate_delta_error(self, goal_position, current_position):
        curr_error = np.linalg.norm(goal_position - current_position)
        self.delta_error = abs(curr_error - self.prev_err)
        self.prev_err = curr_error
       

    def framesForwardKinematics(self, q, joint_ids, goal_positions, base_frame):
        pin.framesForwardKinematics(self.model, self.data, q)
        self.nu.clear()
        for x,joint_id in enumerate(joint_ids):
            iBd = self.data.oMf[base_frame]
            iMl = self.data.oMf[joint_id]
            iMr = iBd.actInv(iMl)
            iMd = iMl.actInv(goal_positions[x])
            self.nu.append(pin.log(iMd).vector)
            self.calculate_delta_error(
                goal_positions[x].translation, iMr.translation)

        return self.nu

    
    def compute_state(self,q, joint_ids, goal_positions, base_frame):
        self.nu.clear()
        self.J_list.clear()
        self.J.fill(0)

        pin.framesForwardKinematics(self.model, self.data, q)
        self.computeFrameJacobian(q)
      

        for x,joint in enumerate(joint_ids):
            iBd = self.data.oMf[base_frame]
            iMl = self.data.oMf[joint]
            iMr = iBd.actInv(iMl)
            iMd = iMl.actInv(goal_positions[x])
            
            if x == 0:
                self.calculate_delta_error(
                    goal_positions[x].translation, iMr.translation)
            
            nu = (pin.log(iMd).vector).reshape(6, 1)

            self.nu.append(nu)
           
            self.J=pin.getFrameJacobian(self.model,self.data,joint,self.fr)
            self.J_list.append(self.J)
        
        return self.nu,self.J_list

    def get_Jpsedo(self, J, J_K):
        J_Kinv = np.linalg.inv(J_K)
        Jp = np.dot(J.T, J_Kinv)
        return Jp

    def find_min(self, A):
        sigma_min = np.min(A)
        if sigma_min < self.epsilon:
            k = ((1 - (sigma_min / self.epsilon)**2)) * self.k_max**2
            k0 = 0
            return k, k0
        else:
            k0 = 10
            k = 0
            return k, k0

    def computeFrameJacobian(self, q):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        
    
    def get_frame_jacobian(self, frame_id):
        self.J.fill(0)
        self.J = pin.getFrameJacobian(self.model,self.data,frame_id,self.fr)
        self.J[:, :6] = 0
        # self.J[3:,:] = 0
        # self.J= pin.getJointJacobian(self.model, self.data, frame_id, self.fr)
        # self.logger.info("Jacobian: {}".format(self.J))
        J_K = np.dot(self.J, self.J.T)
        u, s, vh = np.linalg.svd(J_K)
        sing_values = np.sqrt(s)
        k, k0 = self.find_min(sing_values)
        I = np.identity(self.J.shape[0])
        # self.logger.info("K value: {}".format(k))
        J_damped = np.dot(self.J.T, np.linalg.inv(np.dot(self.J, self.J.T) + k * I))
        return J_damped
    
    def get_damped_jacobian(self,J):    
        J_K = np.dot(J,J.T)
        u, s, vh = np.linalg.svd(J_K)
        sing_values = np.sqrt(s)
        k, k0 = self.find_min(sing_values)
       
        I = np.identity(J.shape[0])
        J_damped = np.dot(J.T, np.linalg.inv(np.dot(J,J.T) + k * I))
        return J_damped

    def get_acceleration(self, q, dq, tau_ref):
        
        pin.aba(self.model, self.data, q, dq, tau_ref)
        return self.data.ddq

    def get_delta_error(self):
        return self.delta_error

    def pd_controller(self, ref_pos, ref_vel, position, velocity):
        Kp = 3000000
        Kd = 300
        
        pos_diff = ref_pos - position
        vel_diff = ref_vel - velocity
        control = Kp * (pos_diff) + Kd * (vel_diff)
        return control

    def compute_recrusive_newtone_euler(self, q, dq, ddq,Fv,B):

        
        M = pin.crba(self.model, self.data, q)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
       
     
        tau = np.dot(M[6:, 6:], ddq) + np.dot(C[6:, 6:], dq[6:]) + np.dot(Fv,dq[6:]) + B + g[6:]
     
        return tau