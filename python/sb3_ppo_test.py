

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv

import unityGymEnv


# we use own replay buffer that saves the observation space as uint8 instead of float32
# int8 is 8bit, float32 is 32bit

n_envs = 10

env_kwargs = {"spawn_point_random": False, "single_goal": False, "frame_stacking": 3, "equalize": True}

# Parallel environments
vec_env = make_vec_env(unityGymEnv.BaseUnityCarEnv, n_envs=n_envs, env_kwargs=env_kwargs)
# the n_envs can quickly be too much since the replay buffer will grow
# the observations are quite big (float32)


n_epochs, batch_size = 5, 64

model = PPO("CnnPolicy", vec_env, verbose=1,
            tensorboard_log="./tmp", n_epochs=n_epochs, batch_size=batch_size)
# CnnPolicy network architecture can be seen in sb3.common.torch_layers.py

modelname="ppo_test_trained_random"
continue_training = False
if continue_training:
    print(f'loading model from file before learning')
    model = PPO.load(modelname, env=vec_env, tensorboard_log="./tmp",
                     n_epochs=n_epochs, batch_size=batch_size)

# TODO model save callback

model.learn(total_timesteps=250000)
model.save(modelname)
