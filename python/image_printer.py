

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym
import time



import gymEnv.carsimGymEnv as carsimGymEnv
import gymEnv.myEnums


import gymEnv.myEnums as myEnums
from gymEnv.myEnums import MapType, LightSetting, SpawnOrientation

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
              "jetBotName": "DifferentialJetBot",
              "spawnOrientation": myEnums.SpawnOrientation.Fixed,
              "frame_stacking": 3, 
              "image_preprocessing": {
                    "downsampling_factor": 2,
                    "grayscale": True,
                    "equalize": True,
                    "contrast_increase": "TODO",
                    "normalize_images": False},
                "coefficients": coefficients,
                "width": 500,
                "height": 168,
                "collisionMode": "oncePerTimestep"}


env = carsimGymEnv.BaseCarsimEnv(**env_kwargs)


# agent preprocessing steps:
env.mapType = carsimGymEnv.MapType.hardBlueFirstLeft
env.log = True
env.reset()
env.getObservation()
env.log = False


# arena screenshots:
# easy eval parcour

env.reset(mapType = MapType.easyBlueFirst, lightSetting=LightSetting.standard)
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("latex_images/evaluation_easy.png")


# medium eval parcour
env.reset(mapType = MapType.mediumBlueFirstLeft, lightSetting=LightSetting.standard)
time.sleep(1) # wait for the car to spawn
env.get_arena_screenshot("latex_images/evaluation_medium.png")

# hard eval parcour
env.reset(mapType = MapType.hardBlueFirstLeft, lightSetting=LightSetting.standard)
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
name= "dark"
time.sleep(1) # wait for the car to spawn
obs_to_file(env.getObservation(), f"latex_images/light_setting_{name}_pov.png")
env.saveObservationNoPreprocessing(f"latex_images/light_setting_{name}_pov_no_preprocessing.png")
env.get_arena_screenshot(f"latex_images/light_setting_{name}_arena.png")

saveAugmentedImages(env, name)


# agent vision with increased lighting
lighting = myEnums.LightSetting.bright
env.reset(lightSetting=lighting)
name = "bright"
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






# agent spawnOrientation

env.reset(mapType = MapType.hardBlueFirstLeft, lightSetting=LightSetting.standard, spawnRot=0.0)
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_Fixed_min.png")

env.reset(mapType = MapType.hardBlueFirstLeft, lightSetting=LightSetting.standard, spawnRot=SpawnOrientation.getOrientationRange(SpawnOrientation.Random)[0])
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_Random_min.png")

env.reset(mapType = MapType.hardBlueFirstLeft, lightSetting=LightSetting.standard, spawnRot=SpawnOrientation.getOrientationRange(SpawnOrientation.VeryRandom)[0])
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_VeryRandom_min.png")


env.reset(mapType = MapType.hardBlueFirstRight, lightSetting=LightSetting.standard, spawnRot=0.0)
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_Fixed_max.png")
env.saveObservationNoPreprocessing(f"latex_images/spawnOrientation_Fixed_max_pov.png")

env.reset(mapType = MapType.hardBlueFirstRight, lightSetting=LightSetting.standard, spawnRot=SpawnOrientation.getOrientationRange(SpawnOrientation.Random)[1])
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_Random_max.png")
env.saveObservationNoPreprocessing(f"latex_images/spawnOrientation_Random_max_pov.png")

env.reset(mapType = MapType.hardBlueFirstRight, lightSetting=LightSetting.standard, spawnRot=SpawnOrientation.getOrientationRange(SpawnOrientation.VeryRandom)[1])
time.sleep(1) # wait for the car to spawn
env.get_arena_topview("latex_images/spawnOrientation_VeryRandom_max.png")
env.saveObservationNoPreprocessing(f"latex_images/spawnOrientation_VeryRandom_max_pov.png")




# images of histograms
filename = "latex_images/preprocessingSteps/fixedSpawnPoint_hardBlueFirstLeft_standard_grayscale.png"
from PIL import Image
image = Image.open(filename)


from gymEnv.histogram_equilization import hist_eq
import numpy as np

image = np.array(image)
# print(f'image shape {image.shape}')

pixels_equalized, histOrig, histEq = hist_eq(image)

pixels_equalized_uint8 = pixels_equalized.astype(np.uint8)

if not os.path.exists("latex_images/histogram"):
    os.mkdir("latex_images/histogram")

env.saveImageGrayscale(image, "latex_images/histogram/original_image.png")
env.saveImageGrayscale(pixels_equalized_uint8, "latex_images/histogram/equalized_image.png")


def save_histogram(img, filename):

    hist, bins = np.histogram(img.flatten(), 256, [0, 255])
    # print(f'hist shape: {hist.shape}')
    # print(f'bins shape: {bins.shape}')
    # print(f'hist: {hist}')
    # print(f'bins: {bins}')

    cdf = hist.cumsum()


    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.bar(bins[:-1], hist, width=1, label="Histogram", color="red")
    ax1.set_ylabel("Histogram")

    ax2 = ax1.twinx()

    # TODO insert the cdf in graph
    # https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py
    
    ax2.plot(range(256), cdf, "k--", linewidth=1.5, label="Cumulative Histogram")
    ax2.set_ylabel("Cumulative Histogram")

    fig.tight_layout()
    plt.savefig(filename)

    plt.close()

save_histogram(image, "latex_images/histogram/original_histogram.png")
save_histogram(pixels_equalized_uint8, "latex_images/histogram/equalized_histogram.png")