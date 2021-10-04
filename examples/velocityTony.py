"""Script demonstrating the joint use of velocity input.

The simulation is run by a `VelocityAviary` environment.

Example
-------
In a terminal, run as:

    $ python velocity.py

Notes
-----
The drones use interal PID control to track a target velocity.

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import random as rand

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary


from stable_baselines3 import  A2C, PPO, TD3
from stable_baselines3.ppo import MlpPolicy
#from stable_baselines3.a2c import MlpPolicy
#from stable_baselines3.td3 import MlpPolicy
#from stable_baselines import  PPO2
#from stable_baselines.common.policies import MlpLstmPolicy
#from stable_baselines.common import make_vec_env
from stable_baselines3.common.env_checker import check_env
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Velocity control example using VelocityAviary')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=2,          type=int,           help='Number of Drones used for the simulation', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=30,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--goal_radius',        default=0.1,        type=float,         help='Radius of the goal (default: 0.1 m)', metavar='')
    parser.add_argument('--cpu',                default=1,          type=int,           help='Number of CPU cores', metavar='')
    parser.add_argument('--collision_time',     default=20,         type=float,         help='Time for the ownship to reach the collision location', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    GOAL_XYZ = np.array([-7,4,6])#[0,rand.randint(-2,2),rand.randint(2,4)])
    COLLISION_POINT = np.array([0,0,6])
    protected_radius = 1
    neighbourhood_radius = 10


    # First row is onwship, second row is intruder
    a = rand.randint(1,1)
    if a == 1:
        x_i = [19,15,15]
    elif a == 2:
        x_i = [7,-15,2]
    elif a == 3:
        x_i = [0,0,1]
    elif a == 4:
        x_i = [0,0,20]
    
    #x_i = rand.uniform(8,15), rand.uniform(-9,9), rand.uniform(1,14)
        #COLLISION_POINT

    x_o = np.array([-10,0,6]) 

    INIT_XYZS = np.vstack((x_o,x_i))

    # Initial attitude of the ownship and intruder
    INIT_RPYS = np.array([
                          [0, 0, 0],
                          [0, 0, 0],
                          ])


    

    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=neighbourhood_radius,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui,
                         goal_xyz=GOAL_XYZ,
                         collision_point = COLLISION_POINT,
                         protected_radius=protected_radius,
                         goal_radius = ARGS.goal_radius,
                         collision_time = ARGS.collision_time,
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()
    
    check_env(env,
              warn=True,
              skip_render_check=True
              )

    #### Compute number of control steps in the simlation ######
    PERIOD = ARGS.duration_sec
    NUM_WP = ARGS.control_freq_hz*PERIOD


    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    START = time.time()

    #Start the model learning
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], qf=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256])])
    #policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], qf=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256])) #For TD3, SAC, DDPG                    
    model = PPO(MlpPolicy,
                env,
                verbose=1,
                tensorboard_log="./ppo_drone_tensorboard/",
                policy_kwargs=policy_kwargs
                )

    #Deeper NN 
    model = PPO.load("PPO", env=env)
    model.learn(total_timesteps=400_000) # Typically not enough
    model.save("PPO")
    model = PPO.load("PPO", env=env)
    #model = PPO.load("PPO_BEST_By_FAR", env=env)

    logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )
    obs = env.reset()
    start = time.time()
    n_trial = 0 
    for i in range(ARGS.duration_sec*env.SIM_FREQ):
        if obs[-1] < neighbourhood_radius : 
            action, _states = model.predict(obs,
                                            deterministic=True,
                                            )
        else:
            action = np.array([1,0,0]) #No Turn

        #print(f"action {action}")
        obs, reward, done, info = env.step(action)

        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state = env._getDroneStateVector(int(j)),
                       control= np.zeros(12)
                       )
        if i%env.SIM_FREQ == 0:
            env.render()
            #print(f"Episode is done: {done}")
        sync(i, start, env.TIMESTEP)
        if done:
            n_trial+=1
            obs = env.reset()
            print(f"Run # {n_trial}")
    env.close()
    logger.save()
    logger.plot()

