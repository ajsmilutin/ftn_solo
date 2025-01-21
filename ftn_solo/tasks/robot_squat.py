import numpy as np
from ftn_solo.utils.pinocchio import PinocchioWrapper
from ftn_solo.controllers.rnea import RneAlgorithm
from .task_base import TaskBase
from scipy.interpolate import CubicSpline



class RobotMove(TaskBase):  
    
    def __init__(self,num_joints,robot_version,config_yaml,logger,dt) -> None:    
        super().__init__(num_joints, robot_version, config_yaml)
        self.pin_robot = PinocchioWrapper(robot_version,logger,dt)
        self.joint_controller = RneAlgorithm(num_joints, self.config["joint_controller"],robot_version,logger,dt)
        # self.alfa_walk = [-51.32,-62.05,-58.42,-48.53]
        self.alfa_walk = [0,0,0,0]
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
        # fl,fr,hl,hr = self.get_leg_position_walk(self.alfa_walk,self.z_vectors_walk)

        # step = [fl[0],fr[0],hl[0],hr[0]]
    
        # self.steps.append(step)

        # step = [fl[0],fr[1],hl[1],hr[0]]
        # self.steps.append(step)

        # step = [fl[3],fr[2],hl[3],hr[2]]
        # self.steps.append(step)

        # step = [fl[1],fr[0],hl[0],hr[1]]
        # self.steps.append(step)

        # step = [fl[2],fr[3],hl[2],hr[3]]
        # self.steps.append(step)


        #Generating steps from vector
        # self.logger.info("steps: {}".format(self.steps))

        # for x,step in  enumerate(self.z_vectors_walk):
           
        #     leg_position=self.get_positions(self.alfa_walk[x],step)
        #     self.steps.append(leg_position)

        # return self.steps

        # Total duration of trajectory (can be adjusted)\
        y=0.1469
        # Define waypoints for a smooth arc-like trajectory
        #  np.array([0.196, 0.1469, -0.20]), -0.15Z 0.05X
        t_arc_points = np.array([0, 0.5, 1])
        x_arc_front = np.array([0.140, 0.196, 0.226])  # X positions
        z_arc_front = np.array([-0.20, -0.15, -0.20])  # Z heights (arc)

     
        x_line_front = np.array([0.226, 0.196, 0.140])  # X positions
        z_line_front = np.array([-0.20, -0.20, -0.20])  # Z heights (arc)

       
        x_arc_back = np.array([0.226, 0.196, 0.140])  # X positions
        z_arc_back = np.array([-0.20, -0.15, -0.20])  # Z heights (arc)

     
        x_line_back = np.array([0.140, 0.196, 0.226])  # X positions
        z_line_back = np.array([-0.20, -0.20, -0.20])  # Z heights (arc)

        self.x_arc_front = CubicSpline(t_arc_points, x_arc_front)
        self.z_arc_front = CubicSpline(t_arc_points, z_arc_front)
        self.x_line_front = CubicSpline(t_arc_points, x_line_front)
        self.z_line_front = CubicSpline(t_arc_points, z_line_front)

        self.x_arc_back = CubicSpline(t_arc_points, x_arc_back)
        self.z_arc_back = CubicSpline(t_arc_points, z_arc_back)
        self.x_line_back = CubicSpline(t_arc_points, x_line_back)
        self.z_line_back = CubicSpline(t_arc_points, z_line_back)
    

    def get_trajectory(self,t,T,T2):
        steps= []
        T_total = T + T2
        # Normalize time within [0, T] using modulo
        t_mod = t % T_total
        R_y = np.eye(3)
        # Cubic time scaling
        # Fifth-order polynomial time scaling (quintic time scaling)
        if t_mod <= T:
            s_t = 10 * (t_mod / T)**3 - 15 * (t_mod / T)**4 + 6 * (t_mod / T)**5
            x_l_f = self.x_arc_front(s_t)
            z_l_f = self.z_arc_front(s_t)
            x_r_f = self.x_line_front(s_t)
            z_r_f = self.z_line_front(s_t)

            x_l_b = self.x_line_back(s_t)
            z_l_b = self.z_line_back(s_t)
            x_r_b = self.x_arc_back(s_t)
            z_r_b = self.z_arc_back(s_t)
          
           
        else:
            t_d = t_mod - T
            s_t = 10 * (t_d / T2)**3 - 15 * (t_d / T2)**4 + 6 * (t_d / T2)**5
            x_l_f = self.x_line_front(s_t)
            z_l_f = self.z_line_front(s_t)
            x_r_f = self.x_arc_front(s_t)
            z_r_f = self.z_arc_front(s_t)

            x_l_b = self.x_arc_back(s_t)
            z_l_b = self.z_arc_back(s_t)
            x_r_b = self.x_line_back(s_t)
            z_r_b = self.z_line_back(s_t)
            
       

        #Position

        if self.start == False:
            v1 = np.array([0.246, 0.1469, -0.20])
            v2 = np.array([0.246, -0.1469, -0.20])
            v3 = np.array([-0.246, 0.1469, -0.20])
            v4 = np.array([-0.246, -0.1469, -0.20])
            odmes1 = self.pin_robot.moveSE3(R_y,v1)
            odmes2 = self.pin_robot.moveSE3(R_y,v2)
            odmes3 = self.pin_robot.moveSE3(R_y,v3)
            odmes4 = self.pin_robot.moveSE3(R_y,v4)
            steps = [odmes1,odmes2,odmes3,odmes4]
            self.start=True
        else:       
            v1 = np.array([x_l_f,0.1469,z_l_f])
            v2 = np.array([x_r_f,-0.1469,z_r_f])
            v3 = np.array([-x_l_b,0.1469,z_l_b])
            v4 = np.array([-x_r_b,-0.1469,z_r_b])
            odmes1 = self.pin_robot.moveSE3(R_y,v1)
            odmes2 = self.pin_robot.moveSE3(R_y,v2)
            odmes3 = self.pin_robot.moveSE3(R_y,v3)
            odmes4 = self.pin_robot.moveSE3(R_y,v4)
            steps = [odmes1,odmes2,odmes3,odmes4]
            

        return steps
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
        
        # self.logger.info("t time: {}".format(t))
        # tourques = self.joint_controller.rnea(self.steps[self.step],position,velocity,sensors['attitude'])
        
        tourques = self.joint_controller.rnea(self.get_trajectory(t,0.06,0.06),position,velocity,sensors['attitude'])
        # self.i+=1
        # # if self.joint_controller.get_delta_error() < self.eps:
        # if self.i ==40:
        #     self.step = self.step + 1
        #     self.i=0
        # if self.step == len(self.steps):
        #     self.step = 1
    
        return tourques
                