import pinocchio as pin
import numpy as np
from robot_properties_solo.solo12wrapper import Solo12Config


class PinocchioWrapper(object):
    def __init__(self,robot_version,logger):
        
        # self.resourses = Resources(robot_version)
        self.pin_robot = Solo12Config.buildRobotWrapper()
        self.model = self.pin_robot.model
        self.data = self.model.createData()
        self.logger=logger
        joint_ids = list(range(self.model.njoints))
        self.logger.info(f"Robot has {joint_ids} joints")
        
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
        self.eps    = 1e-4
        self.IT_MAX = 1000
        self.DT     = 0.05
        self.damp   = 0.05
        self.success = False
        self.i = 0
        self.epsilon = 0.0000002
        self.k_max = 0.00000002

        
    
    def mass(self, q):
        return pin.crba(self.model, self.data, q)

    

    def gravity(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q)

    
    def     moveSE3(self,R,T):
        oMdes = pin.SE3(R,T)
        return oMdes

    def pinIntegrate(self,q,v):
        return pin.integrate(self.model,q,v*self.DT)
   
    def framesForwardKinematics(self,q,joint_id,goalPosition,base_frame):
        pin.framesForwardKinematics(self.model, self.data, q)
        iBd=self.data.oMf[base_frame]
        iMl=self.data.oMf[joint_id]
        iMr=iBd.actInv(iMl)
        iMd = iMr.actInv(goalPosition)
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
    
    def pd_controller(self,ref_pos,ref_vel,position,velocity):
        Kp=4
        Kd=0.05
        zeros=np.zeros(6)
        control = Kp * (ref_pos - position) + Kd * (ref_vel - velocity)
        return np.concatenate((zeros,control))


    def compute_recrusive_newtone_euler(self,q,dq,ddq):
        
        M = pin.crba(self.model,self.data,q)
        C = pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        g = pin.computeGeneralizedGravity(self.model,self.data,q)
         
        tau = np.dot(M,ddq) + np.dot(C,dq) + g  
        
        return tau
      
    