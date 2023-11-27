

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv

import unityGymEnv

from myPPO.myPPO import myPPO

# we use own replay buffer that saves the observation space as uint8 instead of float32
# int8 is 8bit, float32 is 32bit

normalize_images = False
# scales the images to 0-1 range
# requires dtype float32

n_envs = 1
asynch = False
# false is of course much faster
# the new modified PPO with the delayed rewards will not require this asynch and will be much faster

env_kwargs = {"imagelog":True, "asynchronous":asynch, "spawn_point_random": False, "single_goal": False, "frame_stacking": 3, "equalize": True, "normalize_images": normalize_images}


env = unityGymEnv.BaseUnityCarEnv(**env_kwargs)

# TODO build a method that takes an arena screenshot


# arena screenshots:
# easy eval parcour

# medium eval parcour

# hard eval parcour

# training parcour

# single goal training parcour



# agent POV screenshots:

# agent vision with standard lighting

# agent vision with reduced lighting

# agent vision with increased lighting

# agent data augmented POV screenshots:
# agent with salt and pepper noise

