import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm
from .task_base import TaskBase



class RobotMove(TaskBase):  
    
    def __init__(self,num_joints,robot_version,config_yaml,logger,dt) -> None:    
        super().__init__(num_joints, robot_version, config_yaml)
        self.pin_robot = PinocchioWrapper(robot_version,logger,dt)
        self.joint_controller = RneAlgorithm(num_joints, self.config["joint_controller"],robot_version,logger,dt)
            
        self.alfa_walk = [-51.32,-62.05,-58.42,-48.53]
        self.z_vectors_walk = [
            np.array([0.196, 0.1469, -0.20]),
            np.array([0.196, 0.1469, -0.15]),
            np.array([0.246, 0.1469, -0.20]),
            np.array([0.126, 0.1469, -0.20]),
        ]

        self.alfa_rotate_y = [-51.32,-62.05,-58.42,-48.53]
        self.alfa_rotate_x = [0,0,-7.67,7.67]
        self.z_vectors_rotate = [
            np.array([0.196, 0.1469, -0.20]),
            np.array([0.196, 0.1469, -0.15]),
            np.array([0.196, 0.1769, -0.20]),
            np.array([0.196, 0.1169, -0.20]),
        ]
                    
        self.steps=[]
        self.step=0
        self.eps = 0.0018
        self.i=0
        self.start=False
        
        self.logger=logger
    def init_pose(self,q,dq):
        
        #Steps for rotating
        # fl,fr,hl,hr = self.get_leg_position_rotate(self.alfa_rotate_y,self.alfa_rotate_x,self.z_vectors_rotate)

        # step = [fl[0],fr[0],hl[0],hr[0]]
        # self.steps.append(step)

        # step = [fl[0],fr[1],hl[1],hr[0]]
        # self.steps.append(step)

        # step = [fl[0],fr[3],hl[3],hr[0]]
        # self.steps.append(step)

        # step = [fl[1],fr[0],hl[0],hr[1]]
        # self.steps.append(step)

     
        




        #Steps for walking
        fl,fr,hl,hr = self.get_leg_position_walk(self.alfa_walk,self.z_vectors_walk)

        step = [fl[0],fr[0],hl[0],hr[0]]
    
        self.steps.append(step)

        step = [fl[0],fr[1],hl[1],hr[0]]
        self.steps.append(step)

        step = [fl[3],fr[2],hl[3],hr[2]]
        self.steps.append(step)

        step = [fl[1],fr[0],hl[0],hr[1]]
        self.steps.append(step)

        step = [fl[2],fr[3],hl[2],hr[3]]
        self.steps.append(step)

        self.logger.info("steps: {}".format(self.steps))

        # for x,step in  enumerate(self.z_vectors_walk):
           
        #     leg_position=self.get_positions(self.alfa_walk[x],step)
        #     self.steps.append(leg_position)

        # return self.steps
    
    def get_leg_position_rotate(self,angle_y,angle_x,t):
        fl =[]
        fr =[]
        hl =[]
        hr =[]
        for x,t in enumerate(self.z_vectors_walk):

            alfa_y=np.radians(angle_y[x])
            alfa_x=np.radians(angle_x[x])
            R_y=np.array([[np.cos(alfa_y),np.sin(alfa_y)*np.sin(alfa_x),np.sin(alfa_y)*np.cos(alfa_x)],
            [ 0,  np.cos(alfa_x), -np.sin(alfa_x)],
            [ -np.sin(alfa_y), np.cos(alfa_y)*np.sin(alfa_x),np.cos(alfa_y)*np.cos(alfa_x)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            fl.append(oMdes)
            
            t[1]=-t[1]
            alfa_y=np.radians(angle_y[x])
            alfa_x=np.radians(angle_x[x])
            R_y=np.array([[np.cos(alfa_y),np.sin(alfa_y)*np.sin(alfa_x),np.sin(alfa_y)*np.cos(alfa_x)],
            [ 0,  np.cos(alfa_x), -np.sin(alfa_x)],
            [ -np.sin(alfa_y), np.cos(alfa_y)*np.sin(alfa_x),np.cos(alfa_y)*np.cos(alfa_x)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            fr.append(oMdes)
        
            alfa_y=np.radians(-angle_y[x])
            alfa_x=np.radians(angle_x[x])
            t[0]=-t[0]
            t[1]=-t[1]
            R_y=np.array([[np.cos(alfa_y),np.sin(alfa_y)*np.sin(alfa_x),np.sin(alfa_y)*np.cos(alfa_x)],
            [ 0,  np.cos(alfa_x), -np.sin(alfa_x)],
            [ -np.sin(alfa_y), np.cos(alfa_y)*np.sin(alfa_x),np.cos(alfa_y)*np.cos(alfa_x)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            hl.append(oMdes)
        
            alfa_y=np.radians(-angle_y[x])
            alfa_x=np.radians(angle_x[x])
            t[1]=-t[1]
            t[0]=t[0]
            R_y=np.array([[np.cos(alfa_y),np.sin(alfa_y)*np.sin(alfa_x),np.sin(alfa_y)*np.cos(alfa_x)],
            [ 0,  np.cos(alfa_x), -np.sin(alfa_x)],
            [ -np.sin(alfa_y), np.cos(alfa_y)*np.sin(alfa_x),np.cos(alfa_y)*np.cos(alfa_x)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            hr.append(oMdes)
        
        return fl,fr,hl,hr
    
    def get_leg_position_walk(self,angle,t):
        fl =[]
        fr =[]
        hl =[]
        hr =[]
        for x,t in enumerate(self.z_vectors_walk):

            alfa=np.radians(angle[x])
            R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
            [ 0,  1, 0],
            [ -np.sin(alfa),  0 ,np.cos(alfa)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            fl.append(oMdes)
            
            t[1]=-t[1]
            alfa=np.radians(angle[x])
            R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
            [ 0,  1, 0],
            [ -np.sin(alfa),  0 ,np.cos(alfa)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            fr.append(oMdes)
        
            alfa=np.radians(-angle[x])
            t[0]=-t[0]
            t[1]=-t[1]
            R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
            [ 0,  1, 0],
            [ -np.sin(alfa),  0 ,np.cos(alfa)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            hl.append(oMdes)
        
            alfa=np.radians(-angle[x])
            t[1]=-t[1]
            t[0]=t[0]
            R_y=np.array([[np.cos(alfa),0,np.sin(alfa)],
            [ 0,  1, 0],
            [ -np.sin(alfa),  0 ,np.cos(alfa)]])
            oMdes=self.pin_robot.moveSE3(R_y,t)
            hr.append(oMdes)
        
        return fl,fr,hl,hr
    
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
        # if self.joint_controller.get_delta_error() < self.eps:
        if self.i ==30:
            self.step = self.step + 1
            self.i=0
        if self.step == len(self.steps):
            self.step = 1
        
    
        return toqrues
                