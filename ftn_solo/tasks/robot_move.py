import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm
from .task_base import TaskBase



class RobotMove(TaskBase):  
    
    def __init__(self,num_joints,robot_version,config_yaml,logger,dt) -> None:    
        super().__init__(num_joints, robot_version, config_yaml)
        self.pin_robot = PinocchioWrapper(robot_version,logger,dt)
        self.joint_controller = RneAlgorithm(num_joints, self.config["joint_controller"],robot_version,logger,dt)
            
        self.alfa = [-0,-0,-0,-0,0]
        self.z_vectors = [
            np.array([0.500, 0.1469, -0.16]),
            np.array([0.196, 0.1469, -0.11]),
            np.array([0.246, 0.1469, -0.16]),
            np.array([0.196, 0.1469, -0.16]),
            np.array([0.146, 0.1469, -0.16]),
        ]
                    
        self.steps=[]
        self.step=0
        self.eps = 0.0018
        self.i=0
        
        self.logger=logger
    def init_pose(self,q,dq):
    
        
        for x,step in  enumerate(self.z_vectors):
            leg_position=self.get_positions(self.alfa[x],step)
            self.steps.append(leg_position)
            
        return self.steps
    
    
    def get_positions(self,angle,t):   

        pos=[]
        
        alfa=np.radians(angle)
        R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
        [ 0,  1, 0],
        [ -np.sin(alfa),  0 ,np.cos(alfa)]])
        oMdes=self.pin_robot.moveSE3(R_y,t)
        pos.append(oMdes)
        
        t[1]=-t[1]
        alfa=np.radians(angle)
        R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
        [ 0,  1, 0],
        [ -np.sin(alfa),  0 ,np.cos(alfa)]])
        oMdes=self.pin_robot.moveSE3(R_y,t)
        pos.append(oMdes)
    
        alfa=np.radians(-angle)
        t[0]=-t[0]
        t[1]=-t[1]
        R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
        [ 0,  1, 0],
        [ -np.sin(alfa),  0 ,np.cos(alfa)]])
        oMdes=self.pin_robot.moveSE3(R_y,t)
        pos.append(oMdes)
    
        alfa=np.radians(-angle)
        t[1]=-t[1]
        t[0]=t[0]
        R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
        [ 0,  1, 0],
        [ -np.sin(alfa),  0 ,np.cos(alfa)]])
        oMdes=self.pin_robot.moveSE3(R_y,t)
        pos.append(oMdes)
            
        return pos
    
    def compute_control(self, t,position, velocity, sensors):
        toqrues = self.joint_controller.rnea(self.steps[self.step],position,velocity,sensors['attitude'])
        self.i+=1
      
        self.logger.info("sensor: {}".format(sensors['touch'][1]))
        if self.i == 500:
            self.step = self.step + 1
            self.i=0
        if self.step == len(self.steps):
            self.step = 0
        
       
        return toqrues
            