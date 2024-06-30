import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm


class RobotMove():  
    
    def __init__(self,num_joints,robot_version,config_yaml,logger) -> None:    
        
        self.pin_robot = PinocchioWrapper(robot_version,logger)
        self.joint_controller = RneAlgorithm(num_joints, config_yaml,robot_version,logger)
            
        self.alfa = [-38.6248,-60,-48.4917,-10.907]
        self.z_vectors = [
            np.array([0.175, 0.1476, -0.25]),
            np.array([0.175, 0.1476, -0.16]),
            np.array([0.175, 0.1476, -0.25]),  
            np.array([0.175, 0.1476, -0.25]),
        ]
                    
        self.steps=[]
        self.step=0
        self.i=0
        
        self.get_logger=logger
    def init_pose(self,q,dq):
    
        
        for x,step in  enumerate(self.z_vectors):
            leg_position=self.get_positions(self.alfa[x],step)
            self.steps.append(leg_position)
            
        return self.steps
    
    
    def get_positions(self,angle,t):   

        pos=[]
        
        for x in range(0,4):
        
            if x == 1:
                t[1]=-t[1]
                alfa=np.radians(angle)
                R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
                [ 0,  1, 0],
                [ -np.sin(alfa),  0 ,np.cos(alfa)]])
                oMdes=self.pin_robot.moveSE3(R_y,t)
            elif x == 2:
                alfa=np.radians(-angle)
                t[0]=-t[0]
                t[1]=-t[1]
                R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
                [ 0,  1, 0],
                [ -np.sin(alfa),  0 ,np.cos(alfa)]])
                oMdes=self.pin_robot.moveSE3(R_y,t)
            elif x == 3:
                alfa=np.radians(-angle)
                t[1]=-t[1]
                t[0]=t[0]
                R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
                [ 0,  1, 0],
                [ -np.sin(alfa),  0 ,np.cos(alfa)]])
                oMdes=self.pin_robot.moveSE3(R_y,t)
            else:
                alfa=np.radians(angle)
                R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
                [ 0,  1, 0],
                [ -np.sin(alfa),  0 ,np.cos(alfa)]])
                oMdes=self.pin_robot.moveSE3(R_y,t)
                
            pos.append(oMdes)
            
        return pos
    
    def compute_control(self, t,position, velocity, sensors):
        self.i=self.i+1
        toqrues = self.joint_controller.rnea(self.steps[self.step],position,velocity,self.get_logger)
        if self.i >= 200:
            self.step = self.step + 1
            self.i=0
        if self.step == 3:
            self.step = 0
        
        return toqrues
        