

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv

import unityGymEnv

env = unityGymEnv.BaseUnityCarEnv()

# Parallel environments
vec_env = make_vec_env(unityGymEnv.BaseUnityCarEnv, n_envs=1)

#vec_env = DummyVecEnv([env])

model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log="./tmp")

continue_training = False
if continue_training:
    print(f'loading model from file before learning')
    model = PPO.load("ppo_test", env=vec_env, tensorboard_log="./tmp")

model.learn(total_timesteps=25000)
model.save("ppo_test")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_test")
print(f'model loaded')

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

    if dones[0]:
        print(info[0])
