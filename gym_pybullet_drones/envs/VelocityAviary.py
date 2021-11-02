import os
import numpy as np
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

class VelocityAviary(BaseAviary):
    """Multi-drone environment class for high-level planning."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=5,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 goal_xyz= None,
                 collision_point = None,
                 protected_radius=None,
                 goal_radius = None,
                 collision_time = None,
                 ):
        """Initialization of an aviary environment for or high-level planning.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        #### Create integrated controllers #########################
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
        elif drone_model == DroneModel.HB:
            self.ctrl = [SimplePIDControl(drone_model=DroneModel.HB) for i in range(num_drones)]
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,                 
                         goal_xyz= goal_xyz,
                         collision_point = collision_point,
                         protected_radius= protected_radius,
                         goal_radius = goal_radius,
                         collision_time = collision_time 
                         )
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.5*self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment, a turn angle.

        Returns
        -------
        ndarray
            A Box(1,) where the entry is a numpy array

        """
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        #act_lower_bound = np.array([-np.pi/2])
        #act_upper_bound = np.array([ np.pi/2])
        #return spaces.Dict({str(i): spaces.Box(low=act_lower_bound,
        #                                       high=act_upper_bound,
        #                                       dtype=np.float32
        #                                       ) for i in range(self.NUM_DRONES)})



        act_lower_bound = np.array([-1, -1, 0])
        act_upper_bound = np.array([1,  1, 0])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32)
    

        #actions = np.array([i for i in range(-90,93,3)])*np.pi/180

        #return spaces.Discrete(3)

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A box of shape Box{6+7*NUM_INTRUDERS,}

        """




        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        #obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
        #obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        #return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
        #                                                             high=obs_upper_bound,
        #                                                             dtype=np.float32
        #                                                             ),
        #                                         "neighbors": spaces.MultiBinary(self.NUM_DRONES)
        #                                         }) for i in range(self.NUM_DRONES)})


        #Hard coded for only 1 er 

        #observation vector           x         y        vx      vy     x_i         y_i    vx_i       vy_i  doi - rpz     D2GX  d2gy 
        #obs_lower_bound = np.array([-20.,       -20.,  -10,       -10,  -20,       -20,    -10,       -10,   -10,         -40, -40])
        #obs_upper_bound = np.array([ 20.,        20.,   10,        10,   20,        20,     10,        10,   40,           40,  40 ])

        #Aouf
        #observation vector           x         y    doi   doi    doi   doi  D2GX  d2gy 
        obs_lower_bound = np.array([-40.,     -40.,  -10,  -10,   -10,  -10,  -40, -40])
        obs_upper_bound = np.array([ 40.,      40.,   50,   50,    50,   50,   40,  40 ])


        ############################## doi      turn_upper, turn_lower        
        #obs_lower_bound = np.array([  0.,      -np.pi/2,    -np.pi/2])
        #obs_upper_bound = np.array([  20,       np.pi/2,     np.pi/2])

        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float32
                          )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        uses the ownship pos and velocities, relative positions and velocities wrt to the intruder and 
        distance2intruder as well as distance2goal

        TODO : Implement for multiple two intruders

        Returns
        -------
            ndarray of size 6 + 8*NUM_DRONES

        """


        rel_pos = self.pos[1,:]-self.pos[0,:]
        rel_vel = self.vel[1,:]-self.vel[0,:]
        doi = np.linalg.norm(rel_pos)
        d2g = self.pos[0,:]-self.GOAL_XYZ
        normd2g = np.linalg.norm(d2g)


        INIT_VXVYVZ = (self.COLLISION_POINT - self.INIT_XYZS)/10
        turn_angle = np.abs(np.arccos(np.dot(INIT_VXVYVZ[0]/np.linalg.norm(INIT_VXVYVZ[0]),self.vel[0]/np.linalg.norm(self.vel[0]))))

        if hasattr(self, 'turn_upper'):
            pass
        else:
            self.turn_upper = 0

        if hasattr(self, 'turn_lower'):
            pass
        else:
            self.turn_lower = 0

        #obs_vector = np.hstack([doi, self.turn_upper, self.turn_lower])
        #return obs_vector.reshape(3)

        doi1 = np.linalg.norm(self.pos[1,0:2]-self.pos[0,0:2]) - self.PROTECTED_RADIUS
        doi2 = np.linalg.norm(self.pos[2,0:2]-self.pos[0,0:2]) - self.PROTECTED_RADIUS
        doi3 = np.linalg.norm(self.pos[3,0:2]-self.pos[0,0:2]) - self.PROTECTED_RADIUS
        doi4 = np.linalg.norm(self.pos[4,0:2]-self.pos[0,0:2]) - self.PROTECTED_RADIUS

        obs_vector = np.hstack([self.pos[0,0:2],doi1,doi2,doi3,doi4, d2g[0:2]])
        return obs_vector.reshape(8)

        #adjacency_mat = self._getAdjacencyMatrix()
        #return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.
        Converts a dictionary into a 2D array.

        Parameters
        ----------
        action : ndarray
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

        #TODO - Retreive the duration in self.
        
        SPEED_LIMIT = 0.5*self.MAX_SPEED_KMH * (1000/3600)

        ownship_vel = (self.COLLISION_POINT - self.INIT_XYZS[0])/ np.linalg.norm(self.COLLISION_POINT - self.INIT_XYZS[0])
        intruder_vel = (self.COLLISION_POINT - self.INIT_XYZS[1])/ np.linalg.norm(self.COLLISION_POINT - self.INIT_XYZS[1])
        #intruder_vel = [0,0,0] # non moving  boy
        
        unit_vector_vxvyvz = np.vstack((ownship_vel,intruder_vel))


        speed_ratio = np.empty([self.NUM_DRONES,1])
        for i in range(self.NUM_DRONES):
            speed_ratio[i] =np.linalg.norm(self.COLLISION_POINT-self.INIT_XYZS[i])/(SPEED_LIMIT*self.COLLISION_TIME)
        
        #self.target_vel = unit_vector_vxvyvz[0]*speed_ratio[0]*SPEED_LIMIT 
        INIT_VXVYVZ2 = np.hstack((unit_vector_vxvyvz,speed_ratio))
        adjency_mat = self._getAdjacencyMatrix()        
        
        '''
        INIT_VXVYVZ = (self.COLLISION_POINT - self.INIT_XYZS)/ np.linalg.norm(self.GOAL_XYZ - self.INIT_XYZS)
        speed_ratio = np.empty([self.NUINIT_VXVYVZM_DRONES,1])
        for i in range(self.NUM_DRONES):
            speed_ratio[i] =np.linalg.norm(INIT_VXVYVZ[i])/(self.SPEED_LIMIT*self.COLLISION_TIME)
        '''   


        '''
        V_INTRUDER = np.delete(INIT_VXVYVZ2,0,0)
        def compute_elevation_and_azimuth(x1,x2):
            #Find the relative position 
            delta_x = x2[0]-x1[0]
            delta_y = x2[1]-x1[1]
            delta_z = x2[2]-x1[2] 
            xy_norm=np.sqrt(delta_x ** 2 + delta_y ** 2)
            theta = np.arctan(-delta_z/xy_norm) # because you align thumb with y
            psi = np.arctan(delta_y/delta_x)
            return theta,psi
        def cpm(x):
            # Takes a vector (np array) and its associated  cross product matrix 
            return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
        theta,psi = compute_elevation_and_azimuth(self.pos[0],self.pos[1])
        #Compute optimal plane 
        #_, self.turn_upper,self.turn_lower, _, = self.velocity_obstacle()
        x = np.arctan((np.cos(theta)*np.sin(psi))/np.sin(theta))
        if x<0:
            x = np.pi + x
        n = np.array([0,-np.sin(x),np.cos(x)])
        e = n.reshape(3,1)
        R = e@np.transpose(e) + np.cos(action) * (np.identity(3)- e@np.transpose(e))+np.sin(action)*cpm(n)
        vr = R @ (self.vel[0]) 
        self.vr = vr
        self.target_vel = vr
        self.turn_angle = action #store the turn angle before it gets overwritten 
        v_own  = np.hstack((vr,speed_ratio[0]))
        action = np.vstack((v_own,V_INTRUDER))
        '''


        #
        #np.linalg.norm(self.pos[0,:]-self.GOAL_XYZ)<15

        #int(adjency_mat[0][1])>0 and self.collision_detector()

        if  np.linalg.norm(self.vel[0]-self.target_vel)<0.01 :
            
            #heading = np.arccos(np.dot(np.array([1,0,0]),self.vel[0])/np.linalg.norm(self.vel[0]))
            #if heading > np.pi/2:
            #    heading = -(heading - np.pi/2)
            
            if action==0:
                #go straight
                eulerPsi = 0 
            elif action==1:
                #turn left
                eulerPsi = 3 * (np.pi/180) 
            else :
                #turn right
                eulerPsi = -3 * (np.pi/180)
            
            R_eulerPsi = np.array([[np.cos(eulerPsi),-np.sin(eulerPsi),0],
            [np.sin(eulerPsi),np.cos(eulerPsi),0],
            [0,0,1]])

            self.target_vel = R_eulerPsi@self.vel[0]
            #print(self.GOAL_XYZ)
            print(f"Action : {action}, Target Velocity: {self.target_vel}")
            #self.target_vel = (action/np.linalg.norm(action))*speed_ratio[0]*SPEED_LIMIT 
            #action = np.hstack((np.vstack((action/np.linalg.norm(action),intruder_vel)),speed_ratio))
            action = np.hstack((np.vstack((self.target_vel/np.linalg.norm(self.target_vel),intruder_vel)),speed_ratio))
        else:
            try:
                action = np.hstack((np.vstack((self.target_vel,intruder_vel)),speed_ratio))
            except:
                action = np.hstack((np.vstack((ownship_vel,intruder_vel)),speed_ratio))

        rpm = np.zeros((self.NUM_DRONES, 4))


        for k, v in enumerate(action):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(int(k))
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(v[0:3]) != 0:
                v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3], # same as the current position
                                                    target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                    target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
            rpm[int(k),:] = temp
        return rpm

    ################################################################################

    def _computeReward(self,action):
        """Computes the current reward value(s).

        Adds reward by evaluating the inverse of the distance to the goal 
        and substract rewards by the inverse of the distance to the intruder

        Returns
        -------
        Float
            Computed Reward.

        """
        
        adjency_mat = self._getAdjacencyMatrix()
        INIT_VXVYVZ = (self.COLLISION_POINT - self.INIT_XYZS)/self.COLLISION_TIME #/ np.linalg.norm(self.GOAL_XYZ - self.INIT_XYZS)
        
        
        if np.linalg.norm(self.pos[0]-self.pos[1])< (1 * self.PROTECTED_RADIUS):
            bInside = -10
        else:
            bInside = 0

        if np.linalg.norm(self.pos[0]-self.GOAL_XYZ)< (1.5 * self.PROTECTED_RADIUS):
            bGoal = 10
        else:
            bGoal = 0

        

        if True : #int(adjency_mat[0][1])>0 : #and self.collision_detector(): # and np.linalg.norm(self.vel[0]-INIT_VXVYVZ[0])<1e-2 and hasattr(self,'vr') or np.linalg.norm(self.pos[0]-self.pos[1])< (1.05 * self.PROTECTED_RADIUS):

            rel_pos = self.pos[1,:]-self.pos[0,:]
            doi = np.linalg.norm(rel_pos)
            d2g = np.linalg.norm(self.GOAL_XYZ-self.pos[0,:])



            if np.dot(self.vel[0,1:3],self.vel[1,1:3])<-1e-2:
                incentive = 1
            elif np.dot(self.vel[0,1:3],self.vel[1,1:3])>1e-2:
                incentive = -1
            else:
                incentive = 0
            
            
            if np.dot(self.vel[1],self.pos[1]-self.pos[0])>0:
                goodjob = 5
            else:
                goodjob = 0


            rad2deg = 180/np.pi


            

            dir_vector = (self.COLLISION_POINT-self.INIT_XYZS[0,:])/np.linalg.norm(self.COLLISION_POINT-self.INIT_XYZS[0,:])
            n = np.cross(dir_vector,self.vel[0]/np.linalg.norm(self.vel[0]))
            
            if n[-1]>0:
                turn_angle_sign = 1
            else:
                turn_angle_sign = -1

            #turn_angle = turn_angle_sign * self.turn_angle[0] #* np.arccos(np.dot(dir_vector,self.vel[0]/np.linalg.norm(self.vel[0])))
            

            #if 1/turn_angle > 10:
            #    clipped_angle = 1.0
            #else:
            #    clipped_angle = 0.1/turn_angle

            #_,turn_upper, turn_lower,_ = self.velocity_obstacle()

            d = -0.1*np.linalg.norm(np.cross((self.INIT_XYZS[0,:]-self.pos[0,:]),dir_vector))/np.linalg.norm(dir_vector)


            angle_penalty = 0
            awards_turn_angle = 0
            '''
            if turn_angle_sign>0:
                if hasattr(self, 'turn_upper'):
                    if turn_angle < self.turn_upper:
                        angle_penalty += -1.0
                    elif  np.abs(self.turn_upper)*rad2deg <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_upper)*rad2deg + 5  :
                        awards_turn_angle = 8.
                    elif  np.abs(self.turn_upper)*rad2deg +5 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_upper)*rad2deg + 10  :
                        awards_turn_angle = 4.
                    elif  np.abs(self.turn_upper)*rad2deg +10 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_upper)*rad2deg + 15  :
                        awards_turn_angle = 2.
                    elif  np.abs(self.turn_upper)*rad2deg +15 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_upper)*rad2deg + 20  :
                        awards_turn_angle = 1.
                    else:
                        awards_turn_angle = 0.
            else:
                if hasattr(self, 'turn_lower'):
                    if turn_angle<0 and turn_angle > self.turn_lower:
                        angle_penalty += -1.
                    elif  np.abs(self.turn_lower)*rad2deg <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_lower)*rad2deg + 5  :
                        awards_turn_angle = 8. 
                    elif  np.abs(self.turn_lower)*rad2deg +5 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_lower)*rad2deg + 10  :
                        awards_turn_angle = 4.
                    elif  np.abs(self.turn_lower)*rad2deg +10 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_lower)*rad2deg + 15  :
                        awards_turn_angle = 2.
                    elif  np.abs(self.turn_lower)*rad2deg +15 <  np.abs(turn_angle)*rad2deg < np.abs(self.turn_lower)*rad2deg + 20  :
                        awards_turn_angle = 1.
                    else:
                        awards_turn_angle = 0.
            '''
    
            deviation = -10*np.linalg.norm(self.target_vel - self.vel[0])
            forward_bias = np.dot(self.vel[0]/np.linalg.norm(self.vel[0]),np.array([1,0,0]))

            if self.pos[0,2]<0.5:
                bGround=-100
            else:
                bGround = 0

            
            #Abhik term

            #sigma = 5
            
            #abhik = np.min([1,(doi**2-self.PROTECTED_RADIUS**2)/(sigma-self.PROTECTED_RADIUS)])

            if action == 0:
                abhik_forward = 0.5
            else:
                abhik_forward = 0
                
            ## Nabil Aouf paper Explainable Deep Reinforcement Learning for UAV Autonomous Navigation
            C = 0
            Rgoal = np.linalg.norm(self.last_observation[0:2]-self.GOAL_XYZ[0:2]) - np.linalg.norm(self.pos[0,0:2]-self.GOAL_XYZ[0:2]) - C


            reward  = Rgoal + bGoal # + deviation  #+ goodjob + 0.1*doi #+ abhik #-2/doi #awards_turn_angle +  bInside + angle_penalty

            #reward =  - 1/doi + bGoal + bGround  # + 1/(d2g*d2g)
            #reward  = forward_bias + goodjob + bInside - 1000/doi + bGround
            #reward = - np.abs(self.rpy[0, 2]) + np.dot(dir_vector,self.vel[0]) + incentive + bInside #+ 5/d2g #- 1/doi  # - d + 2*doi #- 10*np.linalg.norm(self.vel[0,:]-np.array([1,0,0]))#+ 10/d2g #- 1 /doi
            #np.dot(self.vel[0,:]/np.linalg.norm(self.vel[0,:]),dir_vector) - 1/doi - np.abs(self.rpy[0, 2]) + -10*(self.pos[0,2]-2)**2 + 10

            precision = 4
            #print(f"TotalReward {reward:.{precision}} \t d {10*d:.{precision}} \t yaw {np.abs(self.rpy[0, 2]):.{precision}} \t Incentive {incentive} \tProj: {15*np.dot(dir_vector,self.vel[0]):.{precision}} \t VelOwnY {self.vel[0,1]:.{precision}} \t VelOwnY {self.vel[0,2]:.{precision}}  \t VelIntY{self.vel[1,1]:.{precision}} \t  VelIntZ{self.vel[1,2]:.{precision}}")
            #print(f"TotalReward {reward}, Turn_Angle_deg {rad2deg*turn_angle:.{precision}}, bInside {bInside}, Turn_upper {self.turn_upper*rad2deg:.{precision}}, Turn_lower {self.turn_lower*rad2deg:.{precision}}")
            #print(f"TotalReward {reward:.{precision}} \t forward_bias {forward_bias:.{precision}} \t bInside {bInside} \t deviation {deviation:.{precision}}, abhik {abhik:.{precision}}")
            
            #print(f"Reward {reward}, \t d2g {1/d2g} \t 1/doi {1/doi}")
            #print(f"Reward {reward}, \t Rgoal {np.linalg.norm(self.last_observation[0:2]-self.GOAL_XYZ[0:2]) - np.linalg.norm(self.pos[0,0:2]-self.GOAL_XYZ[0:2])}, \t d2g {d2g}")
            return reward
        
        else:
            return 0

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Checks if the ownship and the intruder is within the protected radius.
        Also checks if the ownship reached the goal

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        protected_rad_gain = [25/2,8,5,10]
        #Checks for a collision
        for j in range(self.NUM_DRONES):
            ownship  = self._getDroneStateVector(int(0))
            intruder = self._getDroneStateVector(int(j))

        
            #Don't compute the distance between the ownship and itself
            if j==0:
                pass
            elif np.linalg.norm(ownship[0:3]-intruder[0:3])<self.PROTECTED_RADIUS*protected_rad_gain[j]:
                print('Crash')
                self.nCrash +=1
                return True

        
        #Check if the ownship is on the ground
        if ownship[2]<0.1:
            print('Hit the ground')
            self.nHitTheGround += 1
            return True


        #Check if the ownship is outside the domain
        if np.linalg.norm(ownship[0:3]-self.GOAL_XYZ) > 40:
            print('Outside the domain')
            self.nOutside += 1
            return True


        #Check if the ownship reached the goal
        if np.linalg.norm(ownship[0:3]-self.GOAL_XYZ) < 1*self.PROTECTED_RADIUS:
            print('Reached the goal')
            self.nGoalReached +=1
            return True

        dir_vector = (self.GOAL_XYZ-self.INIT_XYZS[0,:])/np.linalg.norm(self.GOAL_XYZ-self.INIT_XYZS[0,:])
        n = np.cross(dir_vector,self.vel[0]/np.linalg.norm(self.vel[0]))
        if n[-1]>0:
            turn_angle_sign = 1
        else:
            turn_angle_sign = -1
        turn_angle = turn_angle_sign * np.arccos(np.dot(dir_vector,self.vel[0]/np.linalg.norm(self.vel[0])))

        #if np.linalg.norm(self.pos[0]-self.pos[1])<4:
        #    if np.abs(turn_angle) < np.abs(self.turn_upper) or np.abs(turn_angle) < np.abs(self.turn_lower):
        #        return True


        #if turn_angle > 0 and turn_angle > (self.turn_upper + np.pi/2)/2:
        #    return True
        #
        #if turn_angle < 0 and turn_angle < (self.turn_lower - np.pi/2)/2:
        #    return True

        #Check for the length of the simulation
        if self.step_counter/self.SIM_FREQ > 50:
            print('Times up!')
            self.nTimeout +=1
            return True
        else:
            return False

        
        

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """

        totalRuns = (self.nHitTheGround+self.nOutside + self.nCrash + self.nTimeout + self.nGoalReached)
        info = {"nHitTheGround":self.nHitTheGround, 
                "nOutside":self.nOutside, 
                "nCrash":self.nCrash, 
                "nTimeouts":self.nTimeout, 
                "nGoalReached":self.nGoalReached,
                "Total Runs": totalRuns}
        #return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
        return info
################################################################################
    def collision_detector(self):

        '''Computes if the ownship will collide with the intruder

        Returns
        -------
        bool 

        '''

        #Extract the states for the ownship and intruder
        x1 = self.pos[0,:]
        x2 = self.pos[1,:]
        v1 = self.vel[0,:]
        v2 = self.vel[1,:]

        #Compute the cone opening angle
        d_oi = np.linalg.norm(x2-x1)
        rpz = self.PROTECTED_RADIUS
        d_vo =(d_oi ** 2 - rpz ** 2) / d_oi
        r_vo = rpz*np.sqrt(d_oi**2-rpz**2)/d_oi
        alpha_vo = np.arctan(r_vo/d_vo)
        Doi = x2-x1

        delta_x = x2[0]-x1[0]
        delta_y = x2[1]-x1[1]
        delta_z = x2[2]-x1[2] 

        xy_norm=np.sqrt(delta_x ** 2 + delta_y ** 2)

        theta = np.arctan(-delta_z/xy_norm) # because you align thumb with y
        psi = np.arctan(delta_y/delta_x)

        Dvo = d_vo*np.array([np.cos(theta)*np.cos(psi),np.cos(theta)*np.sin(psi),-np.sin(theta)]) 

        ctheta = np.dot(v1-v2,Dvo)/(np.linalg.norm(v1-v2)*d_vo)
        if np.abs(ctheta)> np.abs(np.cos(alpha_vo)) and (np.dot(v2,Dvo)<0):
            return True
        else:
            #print('Oh noooo!!')
            return False


################################################################################

    def velocity_obstacle(self):
        #TO DO : Put the all the vectors in the body frame.

        def compute_elevation_and_azimuth(x1,x2):
            #Find the relative position 
            delta_x = x2[0]-x1[0]
            delta_y = x2[1]-x1[1]
            delta_z = x2[2]-x1[2] 

            xy_norm=np.sqrt(delta_x ** 2 + delta_y ** 2)

            theta = np.arctan(-delta_z/xy_norm) # because you align thumb with y
            psi = np.arctan(delta_y/delta_x)

            return theta,psi

        def Ry(eulerTheta):
            '''Canonical Rotation about y axis'''
            R_eulerTheta = np.array([[np.cos(eulerTheta),0,-np.sin(eulerTheta)],
            [0,1,0],
            [np.sin(eulerTheta),0,np.cos(eulerTheta)]])
            return R_eulerTheta

        def Rx(eulerPhi): 
            '''Canonical Rotation about the x axis'''
            R_eulerPhi = np.array([[1,0,0],
            [0,np.cos(eulerPhi),np.sin(eulerPhi)],
            [0,-np.sin(eulerPhi),np.cos(eulerPhi)]])
            return R_eulerPhi

        def Rz(eulerPsi):
            '''Canonical Rotation about the z axis'''
            R_eulerPsi = np.array([[np.cos(eulerPsi),-np.sin(eulerPsi),0],
            [np.sin(eulerPsi),np.cos(eulerPsi),0],
            [0,0,1]])
            return R_eulerPsi

        def cpm(x):
            ''' Takes a vector (np array) and its associated  cross product matrix '''
            return np.array([[0,-x[2],x[1]],
            [x[2],0,-x[0]],
            [-x[1],x[0],0]])


        def calculate_velocity_obstacle(rpz,theta,psi,d_oi):
            d_vo =(d_oi ** 2 - rpz ** 2) / d_oi
            r_vo = rpz*np.sqrt(d_oi**2-rpz**2)/d_oi
            alpha_vo = np.arctan(r_vo/d_vo)
            Dvo = d_vo*np.array([np.cos(theta)*np.cos(psi),np.cos(theta)*np.sin(psi),-np.sin(theta)]) 

            return d_vo, r_vo, alpha_vo, Dvo


        def collision_detector(x1,x2,v1,v2,alpha_vo,d_oi,d_avo):
            flag = 0
            Doi = x2-x1
            ctheta = np.dot(v1-v2,Doi)/(np.linalg.norm(v1-v2)*d_oi)

            if np.abs(ctheta)> np.abs(alpha_vo) and d_oi < d_avo and (np.dot(v2,Doi)<0):
                flag =1
            else:
                flag = 0
            
            #print(f"flag:{flag}")
            
            return flag

        def buffer_velocity(alpha_vo,d_vo,Dvo,v_int_i):

            w = 2
            delta_t = 0.1 
            rvi = np.linalg.norm(v_int_i)*np.sqrt(2*(1-np.cos(w*delta_t)))
            v_int_plus_b = v_int_i - rvi * Dvo/(d_vo*np.sin(alpha_vo))
            Dvo_plus = Dvo + (v_int_i-v_int_plus_b)
            dvo_plus = np.linalg.norm(Dvo_plus)

            return v_int_plus_b, Dvo_plus, dvo_plus
 
        def compute_vr(theta,psi,alpha_vo,x1,x2,v1,v2,flag,Dvo,d_vo,rvo):
            vr = self.vel[0]
            nVertices = 100
            turn_angle = 0
            x = 0
            if flag:
                
                #Compute optimal plane angle (x is the plane angle)
                x = np.arctan((np.cos(theta)*np.sin(psi))/np.sin(theta))
                
                if x<0:
                    x = np.pi + x
                #x = 0 

                #Compute collision avoidance plane normal vector
                n = np.array([0,-np.sin(x),np.cos(x)])


                
                
                #Generate turn circle:
                th =  np.linspace(-np.pi,np.pi,100)
                xcircle = np.linalg.norm(v1)*np.cos(th)
                ycircle = np.linalg.norm(v1)*np.sin(th)

                #Compute conic section flags
                bPlane = (np.abs(v2[2]*np.cos(x)-v2[1]*np.sin(x)))<1e-2
                bHyperbola = np.abs(np.pi/2 - np.abs(np.arccos(np.dot(Dvo/d_vo,n))))<=alpha_vo

                #print(f"bPlane{bPlane}, bHyperbola {bHyperbola}")


                #Logic associated with each case
                if bPlane:
                    u = (x2-x1)/np.linalg.norm(x2-x1)
                    e1 = v1/np.linalg.norm(v1) 
                    matrix_sign_of_u = np.sign(Rx(x)@u)
                    sign_of_u = matrix_sign_of_u[1]
                    angleu = np.arccos(np.dot(u,e1))*sign_of_u

                    if sign_of_u <0 :
                        m_upper = np.tan(angleu - alpha_vo) 
                        m_lower = np.tan(angleu + alpha_vo) 
                    else:
                        m_upper = np.tan(angleu - alpha_vo) 
                        m_lower = np.tan(angleu + alpha_vo) 
                    

                    
                    v2body = Rx(x)@v2
                    px = v2body[0]
                    py = v2body[1]
                    v1norm = np.linalg.norm(v1)

                    th2 = np.linspace(-np.pi,np.pi,nVertices)
                    xcircle2 = np.cos(th) + x2[0]
                    ycircle2 = np.sin(th) + x2[1]

                    


                elif bHyperbola:
                    beta = np.linspace(0,2*np.pi,nVertices)
                    a_c = np.zeros(nVertices)
                    vx = np.zeros(nVertices)
                    vy = np.zeros(nVertices)
                    vz = np.zeros(nVertices)

                    for c1 in range(nVertices):
                        #a_c[c1] = (v2[2]*np.cos(x) - v2[1]*np.sin(x))/(np.cos(x)*(np.sin(theta) + np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) - np.sin(x)*(np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) - np.cos(theta)*np.sin(psi) + np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta)))
                        #vx[c1] = (v2[0] + a_c[c1]*np.cos(psi)*np.cos(theta) + a_c[c1]*np.cos(beta[c1])*np.tan(alpha_vo)*np.sin(psi) - a_c[c1]*np.cos(psi)*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(theta))
                        #vy[c1]  = np.sin(x)*(v2[2] - a_c[c1]*np.sin(theta) - a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.cos(x)*(v2[1] + a_c[c1]*np.cos(theta)*np.sin(psi) - a_c[c1]*np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) - a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta))


                        a_c[c1] = (v2[2]*np.cos(x) - v2[1]*np.sin(x))/(np.cos(x)*(np.sin(theta) - np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.sin(x)*(np.cos(theta)*np.sin(psi) + np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) + np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta)))           
                        vx[c1]  = v2[0] + a_c[c1]*np.cos(psi)*np.cos(theta) - a_c[c1]*np.cos(beta[c1])*np.tan(alpha_vo)*np.sin(psi) + a_c[c1]*np.cos(psi)*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(theta)
                        vy[c1]  = np.sin(x)*(v2[2] - a_c[c1]*np.sin(theta) + a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.cos(x)*(v2[1] + a_c[c1]*np.cos(theta)*np.sin(psi) + a_c[c1]*np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) + a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta))




                    #Tan2020 parameters
                    Avo = v2
                    Avophi = Rx(x)@Avo
                    zaphi = Avophi[2]

                    A = d_vo*(np.cos(x)*np.sin(theta)-np.sin(x)*np.sin(psi)*np.cos(theta))+zaphi
                    B = rvo*np.sin(x)*np.cos(psi)
                    C = rvo*(np.sin(x)*np.sin(theta)*np.sin(psi)+np.cos(x)*np.cos(theta))

                    t1 = np.arccos(A/np.sqrt(B**2+C**2))+np.arctan2(C,B)
                    t2 = -np.arccos(-A/np.sqrt(B**2+C**2))+np.arctan2(C,B)

                    if t2<t1:
                        t2 = t2 + 2*np.pi

                    idxt1=np.argmin(np.abs(beta-t1))
                    idxt2=np.argmin(np.abs(beta-t2))


                    x_arr1 = np.array([Avophi[0],vx[idxt1]])
                    y_arr1 = np.array([Avophi[1],vy[idxt1]])
                    line1 = np.polyfit(x_arr1,y_arr1,1)

                    x_arr2 = np.array([Avophi[0],vx[idxt2]])
                    y_arr2 = np.array([Avophi[1],vy[idxt2]])
                    line2 = np.polyfit(x_arr2,y_arr2,1)


                    #This will probably be missnamed
                    m_upper = np.max([line1[0],line2[0]])
                    m_lower = np.min([line1[0],line2[0]])


                    v2body = Rx(x)@v2
                    px = Avophi[0]
                    py = Avophi[1]
                    v1norm = np.linalg.norm(v1)

                else: # Ellipse
                    beta = np.linspace(0,2*np.pi,nVertices)
                    a_c = np.zeros(nVertices)
                    vx = np.zeros(nVertices)
                    vy = np.zeros(nVertices)
                    vz = np.zeros(nVertices)
                    
                    #Compute the ellipse
                    for c1 in range(nVertices):
                        #a_c[c1] = (v2[2]*np.cos(x) - v2[1]*np.sin(x))/(np.cos(x)*(np.sin(theta) + np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) - np.sin(x)*(np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) - np.cos(theta)*np.sin(psi) + np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta)))
                        #vx[c1] = (v2[0] + a_c[c1]*np.cos(psi)*np.cos(theta) + a_c[c1]*np.cos(beta[c1])*np.tan(alpha_vo)*np.sin(psi) - a_c[c1]*np.cos(psi)*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(theta))
                        #vy[c1]  = np.sin(x)*(v2[2] - a_c[c1]*np.sin(theta) - a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.cos(x)*(v2[1] + a_c[c1]*np.cos(theta)*np.sin(psi) - a_c[c1]*np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) - a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta))
                        
                        a_c[c1] = (v2[2]*np.cos(x) - v2[1]*np.sin(x))/(np.cos(x)*(np.sin(theta) - np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.sin(x)*(np.cos(theta)*np.sin(psi) + np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) + np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta)))           
                        vx[c1]  = v2[0] + a_c[c1]*np.cos(psi)*np.cos(theta) - a_c[c1]*np.cos(beta[c1])*np.tan(alpha_vo)*np.sin(psi) + a_c[c1]*np.cos(psi)*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(theta)
                        vy[c1]  = np.sin(x)*(v2[2] - a_c[c1]*np.sin(theta) + a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.cos(theta)) + np.cos(x)*(v2[1] + a_c[c1]*np.cos(theta)*np.sin(psi) + a_c[c1]*np.cos(beta[c1])*np.cos(psi)*np.tan(alpha_vo) + a_c[c1]*np.tan(alpha_vo)*np.sin(beta[c1])*np.sin(psi)*np.sin(theta))


                    #Find the key points of the ellipse     
                    xcenter = np.mean(vx)
                    ycenter = np.mean(vy)

                    center_coords = np.array([xcenter,ycenter])

                    imax = np.argmax(np.sqrt((vx-xcenter)**2 + (vy-ycenter)**2))
                    apex = np.array([vx[imax],vy[imax]])

                    imax2 = np.argmax(np.sqrt((vx[imax]-vx)**2 + (vy[imax]-vy)**2))
                    a2 = np.max(np.sqrt((vx[imax]-vx)**2 + (vy[imax]-vy)**2))
                    a = a2/2

                    xcenter = (vx[imax]+vx[imax2])/2
                    ycenter = (vy[imax]+vy[imax2])/2


                    b = np.min(np.sqrt((vx-xcenter)**2 + (vy-ycenter)**2))

                    tilt_dir = (apex-center_coords)/np.linalg.norm(apex-center_coords)
                    ksi = np.arctan(tilt_dir[1]/tilt_dir[0])


                    #Compute the turn angle
                    Pos_upper = nVertices//2 -1 #Make sure index is an int
                    while (1/a**2)*((xcircle[Pos_upper]-xcenter)*np.cos(ksi)+(ycircle[Pos_upper]-ycenter)*np.sin(ksi))**2 + (1/b**2)*((xcircle[Pos_upper]-xcenter)*np.sin(ksi)-(ycircle[Pos_upper]-ycenter)*np.cos(ksi))**2 <=1.01 : 
                        Pos_upper = Pos_upper+1  
                        if xcircle[Pos_upper]<0 :
                            Pos_upper = nVertices//2 -1 
                            break

                    
                    Pos_lower = nVertices//2 -1  #Make sure index is an int
                    while (1/a**2)*((xcircle[Pos_lower]-xcenter)*np.cos(ksi)+(ycircle[Pos_lower]-ycenter)*np.sin(ksi))**2 + (1/b**2)*((xcircle[Pos_lower]-xcenter)*np.sin(ksi)-(ycircle[Pos_lower]-ycenter)*np.cos(ksi))**2 <=1.01 : 
                        Pos_lower = Pos_lower-1
                        if xcircle[Pos_lower]<0 : 
                            Pos_lower = nVertices//2-1
                            break

                    turn_upper = np.arctan(ycircle[Pos_upper]/xcircle[Pos_upper])
                    turn_lower = np.arctan(ycircle[Pos_lower]/xcircle[Pos_lower])

                    turn_angle_list = np.array([turn_upper,turn_lower])


                    #Select the turn angle
                    turn_angle_abs = np.min(np.abs(turn_angle_list))

                    if np.abs(turn_lower) < 0.05 and np.abs(turn_upper) >0.05:
                        turn_angle = turn_upper
                    elif np.abs(turn_lower)>0.05 and np.abs(turn_upper)<0.05 :
                        turn_angle = turn_lower
                    elif np.abs(turn_lower)<0.05 and np.abs(turn_upper)<0.05 :
                        turn_angle = 0
                    else:
                        if turn_angle_abs//np.abs(turn_upper) ==1:
                            turn_angle = turn_upper
                        else:
                            turn_angle = turn_lower


            if flag:
                k=1 #gain 

                if bHyperbola or bPlane:
                  
                    if (- m_upper**2*px**2 + m_upper**2*v1norm**2 + 2*m_upper*px*py - py**2 + v1norm**2)>=0:
                        sol_upperx = np.array([  ((py + m_upper*(- m_upper**2*px**2 + m_upper**2*v1norm**2 + 2*m_upper*px*py - py**2 + v1norm**2)**(1/2) - m_upper*px)/(m_upper**2 + 1) - py + m_upper*px)/m_upper,
                                        -(py - m_upper*px + (m_upper*(- m_upper**2*px**2 + m_upper**2*v1norm**2 + 2*m_upper*px*py - py**2 + v1norm**2)**(1/2) - py + m_upper*px)/(m_upper**2 + 1))/m_upper])

                        sol_uppery = np.array([(py + m_upper*(- m_upper**2*px**2 + m_upper**2*v1norm**2 + 2*m_upper*px*py - py**2 + v1norm**2)**(1/2) - m_upper*px)/(m_upper**2 + 1),
                                        -(m_upper*(- m_upper**2*px**2 + m_upper**2*v1norm**2 + 2*m_upper*px*py - py**2 + v1norm**2)**(1/2) - py + m_upper*px)/(m_upper**2 + 1)])

                        max_x_upper = np.max(sol_upperx)
                        i_upper = np.argmax(sol_upperx)
                        turn_upper = 1.3*np.abs(np.arctan2(sol_uppery[i_upper],max_x_upper)) #upper is always ccw so 


                    else:
                        sol_upperx = [0,0]
                        sol_uppery = [0,0]
                        turn_upper = np.pi



                    if (- m_lower**2*px**2 + m_lower**2*v1norm**2 + 2*m_lower*px*py - py**2 + v1norm**2) >= 0 :

                        sol_lowerx = np.array([((py + m_lower*(- m_lower**2*px**2 + m_lower**2*v1norm**2 + 2*m_lower*px*py - py**2 + v1norm**2)**(1/2) - m_lower*px)/(m_lower**2 + 1) - py + m_lower*px)/m_lower,
                                    -(py - m_lower*px + (m_lower*(- m_lower**2*px**2 + m_lower**2*v1norm**2 + 2*m_lower*px*py - py**2 + v1norm**2)**(1/2) - py + m_lower*px)/(m_lower**2 + 1))/m_lower])

                        sol_lowery = np.array([ (py + m_lower*(- m_lower**2*px**2 + m_lower**2*v1norm**2 + 2*m_lower*px*py - py**2 + v1norm**2)**(1/2) - m_lower*px)/(m_lower**2 + 1),
                                    -(m_lower*(- m_lower**2*px**2 + m_lower**2*v1norm**2 + 2*m_lower*px*py - py**2 + v1norm**2)**(1/2) - py + m_lower*px)/(m_lower**2 + 1)])

                        max_x_lower = np.max(sol_lowerx)
                        i_lower = np.argmax(sol_lowerx)
                        turn_lower = 1.3*-np.abs(np.arctan2(sol_lowery[i_lower],max_x_lower)) #lower is always cw so positive

                    else :
                        sol_lowerx = [0,0]
                        sol_lowery = [0,0]
                        turn_lower = -np.pi
                    
                    #For imminent scenarios
                    while (- m_upper**2*px**2 + m_upper**2*(k*v1norm)**2 + 2*m_upper*px*py - py**2 + (k*v1norm)**2)<0 or ((np.abs(turn_upper)>np.pi/2 or np.abs(turn_lower)>np.pi/2)) :
                        k=k+1
                        
                        #Compute upper for upper turn
                        sol_upperx = np.array([  ((py + m_upper*(- m_upper**2*px**2 + m_upper**2*(k*v1norm)**2 + 2*m_upper*px*py - py**2 + (k*v1norm)**2)**(1/2) - m_upper*px)/(m_upper**2 + 1) - py + m_upper*px)/m_upper,
                                        -(py - m_upper*px + (m_upper*(- m_upper**2*px**2 + m_upper**2*(k*v1norm)**2 + 2*m_upper*px*py - py**2 + (k*v1norm)**2)**(1/2) - py + m_upper*px)/(m_upper**2 + 1))/m_upper])

                        sol_uppery = np.array([(py + m_upper*(- m_upper**2*px**2 + m_upper**2*(k*v1norm)**2 + 2*m_upper*px*py - py**2 + (k*v1norm)**2)**(1/2) - m_upper*px)/(m_upper**2 + 1),
                                        -(m_upper*(- m_upper**2*px**2 + m_upper**2*(k*v1norm)**2 + 2*m_upper*px*py - py**2 + (k*v1norm)**2)**(1/2) - py + m_upper*px)/(m_upper**2 + 1)])

                        max_x_upper = np.max(sol_upperx)
                        i_upper = np.argmax(sol_upperx)
                        turn_upper = 1.3*np.abs((np.arctan(sol_uppery[i_upper]/max_x_upper))) 
                        turn_upper = np.sign(turn_upper)*(np.abs(turn_upper))#+ np.pi/2)/2

                        #Compute for lower turn
                        sol_lowerx = np.array([((py + m_lower*(- m_lower**2*px**2 + m_lower**2*(k*v1norm)**2 + 2*m_lower*px*py - py**2 + (k*v1norm)**2)**(1/2) - m_lower*px)/(m_lower**2 + 1) - py + m_lower*px)/m_lower,
                                    -(py - m_lower*px + (m_lower*(- m_lower**2*px**2 + m_lower**2*(k*v1norm)**2 + 2*m_lower*px*py - py**2 + (k*v1norm)**2)**(1/2) - py + m_lower*px)/(m_lower**2 + 1))/m_lower])

                        sol_lowery = np.array([ (py + m_lower*(- m_lower**2*px**2 + m_lower**2*(k*v1norm)**2 + 2*m_lower*px*py - py**2 + (k*v1norm)**2)**(1/2) - m_lower*px)/(m_lower**2 + 1),
                                    -(m_lower*(- m_lower**2*px**2 + m_lower**2*(k*v1norm)**2 + 2*m_lower*px*py - py**2 + (k*v1norm)**2)**(1/2) - py + m_lower*px)/(m_lower**2 + 1)])
                        
                        max_x_lower = np.max(sol_lowerx)
                        i_lower = np.argmax(sol_lowerx)
                        turn_lower = -1.3*np.abs((np.arctan(sol_lowery[i_lower]/max_x_lower))) #lower is always cw so positive
                        turn_lower = np.sign(turn_lower)*(np.abs(turn_lower)) #+ np.pi/2)/2

            if flag:
                
                if bPlane or bHyperbola:
                    #Select the turn angle
                    turn_angle_list = np.array([turn_upper,turn_lower])
                    turn_angle_abs = np.min(np.abs(turn_angle_list))

                    if turn_angle_abs//np.abs(turn_upper) ==1:
                        turn_angle = np.abs(turn_upper)
                    else:
                        turn_angle = -np.abs(turn_lower)

                    #print(f"sol_upperx {sol_upperx}, sol_uppery {sol_uppery}, sol_lowerx{sol_lowerx}, sol_lowery{sol_lowery}")
                

                #print(f"turn_upper {turn_upper}, turn_lower {turn_lower}, plane {x}, turn_angle {turn_angle}")
                #print(f"bPlane{bPlane} bHyperbola {bHyperbola} gain k {k}")
                #turn_angle = 0
                e = n.reshape(3,1)
                R = e@np.transpose(e) + np.cos(turn_angle) * (np.identity(3)- e@np.transpose(e))+np.sin(turn_angle)*cpm(n)
                #pass
                vr = R @ (k*v1) 
                #print(f"vr : {vr}")
                

            return vr, turn_upper, turn_lower, x

        

        x_own_i = self.pos[0]
        v_own_i = self.vel[0]

        x_int_i = self.pos[1]
        v_int_i = self.vel[1]

        rpz = self.PROTECTED_RADIUS
        d_avo = self.NEIGHBOURHOOD_RADIUS

        d_oi = np.linalg.norm(x_own_i-x_int_i)


        theta,psi = compute_elevation_and_azimuth(x_own_i,x_int_i)
        d_vo, r_vo, alpha_vo, Dvo = calculate_velocity_obstacle(rpz,theta,psi,d_oi)
        flag = collision_detector(x_own_i,x_int_i,v_own_i,v_int_i,alpha_vo,d_oi,d_avo)
        v_int_i,Dvo,d_vo = buffer_velocity(alpha_vo,d_vo,Dvo,v_int_i)
        vr_b,turn_upper, turn_lower, x = compute_vr(theta,psi,alpha_vo,x_own_i,x_int_i,v_own_i,v_int_i,flag,Dvo,d_vo,r_vo)
        #v_target = vr_b

        return vr_b,turn_upper, turn_lower, x