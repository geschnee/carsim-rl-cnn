

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym
import time

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

env_kwargs = {"asynchronous":asynch, 
              "spawn_point_random": False, "single_goal": False, 
              "frame_stacking": 3, "equalize": True, "normalize_images": normalize_images}


env = unityGymEnv.BaseUnityCarEnv(**env_kwargs)

# TODO build a method that takes an arena screenshot

# agent preprocessing steps:
env.mapType = unityGymEnv.MapType.twoGoalLanesBlueFirstLeftHard
env.log = True
env.reset()
env.getObservation()
env.log = False


# arena screenshots:
# easy eval parcour
env.mapType = unityGymEnv.MapType.easyGoalLaneMiddleBlueFirst
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("expose_images/evaluation_easy.png")


# medium eval parcour
env.mapType = unityGymEnv.MapType.twoGoalLanesBlueFirstLeftMedium
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("expose_images/evaluation_medium.png")

# hard eval parcour
env.mapType = unityGymEnv.MapType.twoGoalLanesBlueFirstLeftHard
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("expose_images/evaluation_hard.png")

# training parcour
env.mapType = unityGymEnv.MapType.random
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("expose_images/training_parcour.png")

# single goal training parcour
env.mapType = unityGymEnv.MapType.random
env.single_goal = True
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("expose_images/training_single.png")

env.single_goal = False

# agent POV screenshots:
import PIL.Image as Image
import data_augmentation as da

def obs_to_file(obs, filename):
    im = Image.fromarray(obs, 'L')
    im.save(filename)

print(f'TODO fix lighting')

print(f'warning maximilian only did tests with ambient lighting, bright and dark')
print(f'the current standard was not used in the tests')
print(f'it looks like the training was done with ambient lighting')

print(f'die Lightquellen sind sehr senkrecht ueber den Objekten, deswegen erkennt man die Farben nicht so gut')

# agent vision with ambient lighting
env.mapType = unityGymEnv.MapType.twoGoalLanesBlueFirstLeftHard
env.reset()
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), "expose_images/light_setting_pov_ambient.png")
env.get_arena_screenshot("expose_images/light_setting_arena_ambient.png")


# agent vision with standard lighting
env.reset()
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), "expose_images/light_setting_pov_standard.png")
env.get_arena_screenshot("expose_images/light_setting_arena_standard.png")


# agent vision with reduced lighting
env.lighting = 0.5
env.reset()
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), "expose_images/light_setting_pov_reduced_lighting.png")
env.get_arena_screenshot("expose_images/light_setting_arena_reduced_lighting.png")



# agent vision with increased lighting
env.lighting = 1.5
env.reset()
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), "expose_images/light_setting_pov_increased_lighting.png")
env.get_arena_screenshot("expose_images/light_setting_arena_increased_lighting.png")

# agent data augmented POV screenshots:
# agent with salt and pepper noise

env.mapType = unityGymEnv.MapType.random
env.lighting = 1 # default lighting
env.reset()
time.sleep(1) # wait for the car to spawn
obs = env.getObservation()
obs_to_file(obs, "expose_images/data_entry_original.png")

obs_salt_pepper = da.salt_and_pepper_noise(obs, prob=0.005)
obs_to_file(obs_salt_pepper, "expose_images/data_entry_augmented_salt_and_pepper.png")

sigma = 5
obs_gaussian = da.gaussian_noise(obs, mean=0, sigma=sigma)
obs_to_file(obs_gaussian, f'expose_images/data_entry_augmented_gaussian_sigma_{sigma}.png')
