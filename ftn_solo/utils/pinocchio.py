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
        self.M = np.array([])
        self.C = np.array([])
        self.G = np.array([])

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
        # self.max_tau = 20.0

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


    def computeFrameJacobian(self, q,dq):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

    def computeNonLinear(self, q, dq):
        self.M = pin.crba(self.model, self.data, q)
        self.C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        self.G = pin.computeGeneralizedGravity(self.model, self.data, q)
    
    
    def get_frame_jacobian(self, frame_id):
        self.J.fill(0)
        J_dot=np.zeros((6, 18))
        self.J = pin.getFrameJacobian(self.model,self.data,frame_id,self.fr)
        J_dot = pin.getFrameJacobianTimeVariation(self.model,self.data,frame_id,self.fr)
        self.J[:, :6] = 0
        self.J[3:,:] = 0
        J_dot[:, :6] = 0
     
        return  self.J, J_dot
    

    def get_acceleration(self, q, dq, tau_ref):
        
        pin.aba(self.model, self.data, q, dq, tau_ref)
        return self.data.ddq

    def get_delta_error(self):
        return self.delta_error
    
    def get_tau_constraint(self,J_real,J_dot,dq):
        h = np.dot(self.C[6:,6:], dq[6: ]) + self.G[6:]
        Jdot_theta = np.dot(J_dot[:,6:], dq[6:])
        J = J_real[:,6:]
        zero_block = np.zeros((J.shape[0], J.shape[0]))
        b = np.concatenate((-h, -Jdot_theta))  
        a = np.block([
            [self.M[6:,6:],J.T],
            [J,zero_block]
        ])

        epsilon = 1e-6  # Small regularization constant
        a_reg = a + epsilon * np.eye(a.shape[0])

       

        solve = np.linalg.solve(a_reg, b)
        ddq = solve[:self.M[6:,6:].shape[0]]
        lamba = solve[self.M[6:,6:].shape[0]:]
        
        tau_constraint = np.dot(J.T, lamba)
        return tau_constraint


    def pd_controller(self, ref_pos, ref_vel, position, velocity):
        Kp = 2000000
        Kd = 200
        
        pos_diff = ref_pos - position
        vel_diff = ref_vel - velocity

        control =  Kp * (pos_diff) + Kd * (vel_diff)
        # return np.concatenate((np.zeros(6),control))
        return control
    
   

    def compute_recrusive_newtone_euler(self, dq, ddq,Fv,B,J,J_dot):


        #Projection matrix 
        # J_M_inv = np.dot(J,np.linalg.inv(self.M[6:, 6:]))

        # lambda_matrix = np.linalg.pinv(np.dot(J_M_inv, J.T))
        # P = np.eye(12) - np.dot(np.dot(J.T, lambda_matrix), J_M_inv)
     
        # h=np.dot(self.C[6:, 6:], dq[6:]) +  self.G[6:]
        # p_ddq = np.dot(P, ddq) 

        # tau = np.dot(P, np.dot(self.M[6:, 6:], ddq) + h)


        #Augmented system 
        h=np.dot(self.C[6:, 6:], dq[6:]) +  self.G[6:]
        zero_block = np.zeros((J.shape[0], J.shape[0]))
        A_aug = np.block([
            [self.M[6:, 6:], J.T],
            [J, zero_block]
            ])
        
        dynamics_rhs = np.dot(self.M[6:, 6:], ddq) + h
        constraint_rhs = -np.dot(J_dot, dq[6:])

        b_aug = np.concatenate([dynamics_rhs, constraint_rhs])

        solution = np.linalg.lstsq(A_aug, b_aug, rcond=1e-6)[0]
        ddq_test = solution[:12]         # Joint accelerations
        lambda_vector = solution[12:]  # Constraint forces

        # tau = np.dot(self.M[6:, 6:], ddq_test) + h + np.dot(J.T, lambda_vector) + np.dot(Fv,dq[6:]) + B   # Add constraint contribution



# 
        tau = np.dot(self.M[6:, 6:], ddq) + np.dot(self.C[6:, 6:], dq[6:]) + np.dot(Fv,dq[6:]) + B + self.G[6:]

        # self.logger.info("tau: {}".format(tau))
        # self.logger.info("tau_test: {}".format(tau_test))

        # return np.clip(tau, -self.max_tau, self.max_tau)
        # return tau[6:]
        return tau