

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv

import gymEnv.carsimGymEnv as carsimGymEnv

from myPPO.myPPO import myPPO

import gymEnv.myEnums as myEnums

import os

if not os.path.exists("latex_images"):
    os.makedirs("latex_images")

# we use own replay buffer that saves the observation space as uint8 instead of float32
# int8 is 8bit, float32 is 32bit

normalize_images = False
# scales the images to 0-1 range
# requires dtype float32

n_envs = 1
fixedTimestepsLength = False
# false --> timestep length is determined by python speed
# some other value --> timestep length is fixed to this value

coefficients = {"distanceCoefficient": 0.5,
    "orientationCoefficient": 0.0,
    "velocityCoefficient": 0.0,
    "eventCoefficient": 1.0}

env_kwargs = {"fixedTimestepsLength": fixedTimestepsLength, 
              "jetbot": "DifferentialJetBot",
              "spawn_point": "Fixed",
              "frame_stacking": 3, 
              "image_preprocessing": {
                    "downsampling_factor": 2,
                    "grayscale": True,
                    "equalize": True,
                    "contrast_increase": "TODO",
                    "normalize_images": False},
                "coefficients": coefficients,
                "width": 500,
                "height": 168}


env = carsimGymEnv.BaseCarsimEnv(**env_kwargs)

# agent preprocessing steps:
env.mapType = carsimGymEnv.MapType.hardBlueFirstLeft
env.log = True
env.reset()
env.getObservation()
env.log = False


# arena screenshots:
# easy eval parcour
env.mapType = carsimGymEnv.MapType.easyBlueFirst
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("latex_images/evaluation_easy.png")


# medium eval parcour
env.mapType = carsimGymEnv.MapType.mediumBlueFirstLeft
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("latex_images/evaluation_medium.png")

# hard eval parcour
env.mapType = carsimGymEnv.MapType.hardBlueFirstLeft
env.reset()
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("latex_images/evaluation_hard.png")


# agent POV screenshots:
import PIL.Image as Image
import gymEnv.data_augmentation as da

def obs_to_file(obs, filename):
    im = Image.fromarray(obs, 'L')
    im.save(filename)


def saveAugmentedImages(env, name):
    # agent data augmented POV screenshots:
    # agent with salt and pepper noise

    obs = env.getObservation()

    obs_salt_pepper = da.salt_and_pepper_noise(obs, prob=0.005)
    obs_to_file(obs_salt_pepper, f"latex_images/light_setting_{name}_pov_augmented_salt_and_pepper.png")

    sigma = 5
    obs_gaussian = da.gaussian_noise(obs, mean=0, sigma=sigma)
    obs_to_file(obs_gaussian, f'latex_images/light_setting_{name}_pov_augmented_gaussian_sigma_{sigma}.png')

# Lighting

# I do not use the ambient light setting
# Maximilian used the ambient light setting for the training and also in one of the evaluations
# instead i only use the directional lights with different intensities

env.mapType = carsimGymEnv.MapType.hardBlueFirstLeft

# TODO images with and without histogram equalization

# agent vision with standard lighting
env.reset(lightSetting=myEnums.LightSetting.standard)
time.sleep(1) # wait for the car to spawn
name = "standard"
obs_to_file(env.getObservation(), f"latex_images/light_setting_pov.png")
env.saveObservationNoPreprocessing(f"latex_images/light_setting_{name}_pov_no_preprocessing.png")
env.get_arena_screenshot(f"latex_images/light_setting_{name}_arena.png")

saveAugmentedImages(env, name)



# agent vision with reduced lighting
lighting = myEnums.LightSetting.dark
env.reset(lightSetting=lighting)
name= "reduced_lighting"
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), f"latex_images/light_setting_{name}_pov.png")
env.saveObservationNoPreprocessing(f"latex_images/light_setting_{name}_pov_no_preprocessing.png")
env.get_arena_screenshot(f"latex_images/light_setting_{name}_arena.png")

saveAugmentedImages(env, name)


# agent vision with increased lighting
lighting = myEnums.LightSetting.bright
env.reset(lightSetting=lighting)
name = "increased_lighting"
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), f"latex_images/light_setting_{name}_pov.png")
env.saveObservationNoPreprocessing(f"latex_images/light_setting_{name}_pov_no_preprocessing.png")
env.get_arena_screenshot(f"latex_images/light_setting_{name}_arena.png")

saveAugmentedImages(env, name)

# save images with same spawn pos and map but different lighting including all preprocessing steps

if not os.path.exists('latex_images/preprocessingSteps/'):
    os.makedirs('latex_images/preprocessingSteps/')

# agent vision with standard lighting
lighting = myEnums.LightSetting.standard
mapType=myEnums.MapType.hardBlueFirstLeft
prefix=f"latex_images/preprocessingSteps/fixedSpawnPoint_hardBlueFirstLeft_standard"
env.reset(lightSetting=lighting, mapType=mapType)
time.sleep(1) # wait for the car to spawn
env.saveObservation(prefix)

# agent vision with reduced lighting
lighting = myEnums.LightSetting.dark
prefix=f"latex_images/preprocessingSteps/fixedSpawnPoint_hardBlueFirstLeft_dark"
env.reset(lightSetting=lighting, mapType=mapType)
time.sleep(1) # wait for the car to spawn
env.saveObservation(prefix)

# agent vision with increased lighting
lighting = myEnums.LightSetting.bright
prefix=f"latex_images/preprocessingSteps/fixedSpawnPoint_hardBlueFirstLeft_bright"
env.reset(lightSetting=lighting, mapType=mapType)
time.sleep(1) # wait for the car to spawn
env.saveObservation(prefix)






