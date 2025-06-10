import pinocchio as pin
import numpy as np
# import casadi as cs
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

        self.fr = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED

        # Parameters for IK
        self.DT = 0.01
        self.epsilon = 0.00000001
        self.k_max = 0.00000002
        self.J = np.zeros((6, 18))
        self.J_list=[]
        self.max_tau = 10.0


    def moveSE3(self, R, T):
        oMdes = pin.SE3(R, T)
        return oMdes

    def pinIntegrate(self, q, v):
        return pin.integrate(self.model, q, v*self.DT)

    

    def frames_forward_kinematics(self, q, dq):
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.forwardKinematics(self.model, self.data, q, dq,0*dq)
        
        # self.nu.clear()
        # for x,joint_id in enumerate(joint_ids):
        #     iMl = self.data.oMf[joint_id]
        #     iMd = iMl.actInv(goal_positions[x])
        #     err = iMd.translation - iMl.translation
            
        #     self.nu.append(pin.log(iMd).vector)
            
        # return self.nu
    
    def calculate_velocity(self,joint_id, goal_positions):
        iMl = self.data.oMf[joint_id]
        iMd = iMl.actInv(goal_positions) 
        vel_frame = pin.getFrameVelocity(self.model, self.data, joint_id, self.fr)
        err = pin.log(iMd).vector
        return vel_frame, iMd.translation, err
    


    def compute_frame_jacobian(self, q,dq):
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

    def compute_non_linear(self, q, dq):
        self.M = pin.crba(self.model, self.data, q)
        self.C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        self.G = pin.computeGeneralizedGravity(self.model, self.data, q)
        
    
    
    def get_frame_jacobian(self, frame_id):
        self.J.fill(0)
        J_dot=np.zeros((6, 18))
        self.J = pin.getFrameJacobian(self.model,self.data,frame_id,self.fr)
        J_dot = pin.getFrameJacobianTimeVariation(self.model,self.data,frame_id,self.fr)
        ades = pin.getFrameAcceleration(self.model, self.data,frame_id,self.fr)
        # self.logger.info("Ades: {}".format(ades))
        
        self.J[:, :6] = 0
      
     
        return  self.J[:3,:], J_dot[:3,:],ades
    

  
    

    def compute_recursive_newton_euler(self, dq, ddq, Fv, B, J, J_dot,tau_g):

        # tau = cs.MX.sym('tau', 12)  # 12 joint torques
        # Fc = cs.MX.sym('Fc', 3)     # 3 contact forces

        # # Cost function: minimize tau
        # cost = cs.sumsqr(tau)

        # # Constraint equations
        # mass = 2.5
        # g = 9.81
        # J = np.array(J)
        # J.fill(0)
        # Fc_gravity = np.array([0, 0, mass * g])  # 3 contact forces (gravity compensation)

        # # Convert numerical variables to CasADi types
        # M = cs.DM(self.M[6:, 6:])  # Mass matrix
        # C = cs.DM(self.C[6:, 6:])  # Coriolis matrix
        # Fv = cs.DM(Fv)            # Viscous friction matrix
        # B = cs.DM(B)              # External forces
        # G = cs.DM(self.G[6:])     # Gravity vector
        # J = cs.DM(J)              # Contact Jacobian
        # ddq = cs.DM(ddq)          # Joint accelerations
        # dq = cs.DM(dq[6:])        # Joint velocities

        # # Dynamics constraint
        # dynamic = (
        #     M @ ddq +  # Inertial forces
        #     C @ dq +   # Coriolis forces
        #     Fv +  # Viscous friction
        #     B +        # External forces
        #     G -        # Gravity
        #     tau -      # Joint torques
        #     J.T @ Fc   # Contact forces
        # )

        # # self.logger.info("Dynamic: {}".format(dynamic.shape))

        # # Torque limits
        # torque_limit = self.max_tau  # Scalar or vector

        # # No-penetration constraint
        # no_penetration_const =  Fc[2]  # Scalar

        # # Formulate NLP problem
        # nlp = {
        #     'x': cs.vertcat(tau, Fc),  # Decision variables (15 elements)
        #     'f': cost,
        #     'g': cs.vertcat(
        #         dynamic,  # Dynamics constraint
        #         no_penetration_const,  # No-penetration constraint
        #         torque_limit  # Torque limits
        #     )
        # }

        # # Initial guess
        # tau_gravity = np.dot((-np.linalg.pinv(J.T)), mass * g)  # Gravity compensation for tau
        # # self.logger.info("Tau gravity: {}".format(tau_gravity.shape))
        # # tau_gravity = tau_gravity.flatten()  # Ensure 1D array (12 elements)
        # initial_guess = np.concatenate([tau_gravity[2][:], Fc_gravity])  # 15 elements
     
        # # self.logger.info("Fc gravirtty: {}".format(Fc_gravity.shape))
        # # Solve the problem
        # solver = cs.nlpsol('solver', 'ipopt', nlp)

        # # self.logger.info("Initial guess: {}".format(initial_guess.shape))

        # result = solver(x0=initial_guess, lbg=0, ubg=0)
        # # Extract solution
        # tau_opt = result['x'][:12]
        # tau = np.array(tau_opt).flatten()
       
        # self.logger.info("tau: {}".format(tau))
        # Fc_opt = result['x'][12:]


        # ========================================================================================================================================
        # Augmented system 
        # h=np.dot(self.C[6:, 6:], dq[6:]) +  self.G[6:]
        # zero_block = np.zeros((J.shape[0], J.shape[0]))
        # A_aug = np.block([
        #     [self.M[6:, 6:], J.T],
        #     [J, zero_block]
        #     ])
        
        # dynamics_rhs = np.dot(self.M[6:, 6:], ddq) + h
        # constraint_rhs = -np.dot(J_dot, dq[6:])

        # b_aug = np.concatenate([dynamics_rhs, constraint_rhs])

        # solution = np.linalg.lstsq(A_aug, b_aug, rcond=1e-6)[0]
        # ddq_test = solution[:12]         # Joint accelerations
        # lambda_vector = solution[12:]  # Constraint forces

        # tau = np.dot(self.M[6:, 6:], ddq_test) + h + np.dot(J.T, lambda_vector) + np.dot(Fv,dq[6:]) + B + tau_g  # Add constraint contribution

        #========================================================================================================================================

        # tau = np.dot(self.M[6:, 6:], ddq) + np.dot(self.C[6:, 6:], dq[6:]) + np.dot(Fv,dq[6:]) + B + self.G[6:] #No constraint contribution
        tau = np.dot(self.M[6:, 6:], ddq) + np.dot(self.C[6:, 6:], dq[6:]) + np.dot(Fv,dq[6:]) + B + self.G[6:] + tau_g  # Add constraint contribution

        # self.logger.info("tau: {}".format(tau))
        # self.logger.info("tau_test: {}".format(tau_test))
        #========================================================================================================================================

        # return tau

        return np.clip(tau, -self.max_tau, self.max_tau)
        # return tau[6:]
        return tau