

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

n_envs = 10
asynch = False
# false is of course much faster
# the new modified PPO with the delayed rewards will not require this asynch and will be much faster

env_kwargs = {"asynchronous":asynch, "spawn_point_random": False, "single_goal": False, "frame_stacking": 3, 
              "equalize": True, "normalize_images": normalize_images,
              "bootstrap_n": 1}

# Parallel environments
vec_env = make_vec_env(unityGymEnv.BaseUnityCarEnv, n_envs=n_envs, env_kwargs=env_kwargs)
# the n_envs can quickly be too much since the replay buffer will grow
# the observations are quite big (float32)


n_epochs, batch_size = 5, 64
n_steps = 128 # amount of steps to collect per epoch

algo = myPPO # or myPPO (handles the ansynchronicity of the envs)


print(f'using {algo} with {n_epochs} epochs, {batch_size} batch size and {n_steps} steps per epoch')
policy_kwargs = {"normalize_images": normalize_images}

model = algo("CnnPolicy", vec_env, verbose=1,
            tensorboard_log="./tmp", n_epochs=n_epochs, batch_size=batch_size, n_steps=n_steps, policy_kwargs=policy_kwargs)
# CnnPolicy network architecture can be seen in sb3.common.torch_layers.py

modelname="ppo_test_trained_random"
continue_training = False
if continue_training:
    print(f'loading model from file before learning')
    model = PPO.load(modelname, env=vec_env, tensorboard_log="./tmp",
                     n_epochs=n_epochs, batch_size=batch_size)

# TODO model save callback

model.learn(total_timesteps=2500000)
model.save(modelname)
