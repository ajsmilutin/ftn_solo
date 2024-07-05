import pinocchio as pin
import numpy as np
from robot_properties_solo.solo12wrapper import Solo12Config


class PinocchioWrapper(object):
    def __init__(self,robot_version,logger,dt):
        
        # self.resourses = Resources(robot_version)
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.model = self.pin_robot.model
        self.data = self.model.createData()
        self.logger=logger
        self.dt=dt
        self.prev_err = np.inf
        self.delta_error = np.inf
                
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
        
        
        self.fr=pin.ReferenceFrame.LOCAL
        
        #Parameters for IK
        self.DT     = 0.02
        self.epsilon = 0.0000002
        self.k_max = 0.00000002

        
    
    def mass(self, q):
        return pin.crba(self.model, self.data, q)

    

    def gravity(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    
    def moveSE3(self,R,T):
        oMdes = pin.SE3(R,T)
        return oMdes

    def pinIntegrate(self,q,v):
        return pin.integrate(self.model,q,v*self.DT)
    
    def calculate_delta_error(self,goal_position,current_position):
        curr_error = np.linalg.norm(goal_position - current_position)
        self.delta_error = abs(curr_error - self.prev_err)
        self.prev_err = curr_error
    
   
    def framesForwardKinematics(self,q,joint_id,goal_position,base_frame,x):
        pin.framesForwardKinematics(self.model, self.data, q)
        iBd=self.data.oMf[base_frame]
        iMl=self.data.oMf[joint_id]
        iMr=iBd.actInv(iMl)
        iMd = iMr.actInv(goal_position)
        if x == 0:
            self.calculate_delta_error(goal_position.translation,iMr.translation)
        nu=pin.log(iMd).vector
        return nu
   
    def get_Jpsedo(self,J,J_K):
        J_Kinv=np.linalg.inv(J_K)
        Jp=np.dot(J.T,J_Kinv)
        return Jp
        
   
    def find_min(self,A):
        sigma_min=np.min(np.diag(A))
        if sigma_min < self.epsilon:
            k=((1 - (sigma_min/ self.epsilon)**2)) * self.k_max**2
            k0=0
            return k,k0
        else:
            k0=10
            k=0
            return k,k0
   
    def computeFrameJacobian(self,q,frame_id):
        J_real = pin.computeFrameJacobian(self.model, self.data,q,frame_id)
        J = J_real[:,6:]
        J_K=np.dot(J,J.T)
        u,s,vh=np.linalg.svd(J_K)
        sing_values=np.sqrt(s)
        zeros=np.zeros((6,6))
        np.fill_diagonal(zeros,sing_values)
        k,k0=self.find_min(zeros)
        I = np.identity(J.shape[0])
        J_damped = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + k * I))
        zeros_matrix = np.zeros((6, 6))
        extended_matrix = np.vstack((zeros_matrix,J_damped))
        
        return extended_matrix
        
    def get_acceleration(self,q,dq,tau_ref):
        pin.aba(self.model,self.data,q,dq,tau_ref)
        return self.data.ddq
    
    def get_delta_error(self):
        return self.delta_error
    
    def pd_controller(self,ref_pos,ref_vel,position,velocity):
        Kp=2.0
        Kd=0.1
        zeros=np.zeros(6)
        control = Kp * (ref_pos - position) + Kd * (ref_vel - velocity)
        return np.concatenate((zeros,control))


    def compute_recrusive_newtone_euler(self,q,dq,ddq):
        
        M = pin.crba(self.model,self.data,q)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        g = pin.computeGeneralizedGravity(self.model,self.data,q)
         
        tau = np.dot(M,ddq) + np.dot(C,dq) + g  
        
        return tau
      
    