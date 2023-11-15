from typing import Any, SupportsFloat
import gymnasium.spaces as spaces
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from stable_baselines3.common.env_checker import check_env as check_env_sb3

import numpy as np

from peaceful_pie.unity_comms import UnityComms

from skimage import color
from skimage.measure import block_reduce

import PIL.Image as Image
import io
import base64

from dataclasses import dataclass

import time

from stable_baselines3.common import torch_layers

from histogram_equilization import hist_eq

@dataclass
class StepReturnObject:
    obs: str
    reward: float
    done: bool
    terminated: bool
    info: dict

from enum import Enum
class MapType(Enum):
    random = 0,
    easyGoalLaneMiddleBlueFirst = 1,
    easyGoalLaneMiddleRedFirst = 2,

    twoGoalLanesBlueFirstLeftMedium = 3,
    twoGoalLanesBlueFirstRightMedium = 4,
    twoGoalLanesRedFirstLeftMedium = 5,
    twoGoalLanesRedFirstRightMedium = 6,

    twoGoalLanesBlueFirstLeftHard = 7,
    twoGoalLanesBlueFirstRightHard = 8,
    twoGoalLanesRedFirstLeftHard = 9,
    twoGoalLanesRedFirstRightHard = 10,


class BaseUnityCarEnv(gym.Env):

    unity_comms: UnityComms = None
    instancenumber = 0

    def __init__(self, width=168.0, height=168, port=9000, log=False, asynchronous=True, spawn_point_random=False, single_goal=False, frame_stacking=5, grayscale=True, normalize_images=False, equalize=False):
        self.equalize = equalize
        self.downsampling = 2

        self.log = log

        self.asynchronous = asynchronous
        self.width = int(width / self.downsampling)
        self.height = int(height / self.downsampling)
        # width and height in pixels of the screen

        # how are RL algos trained for continuous action spaces?
        # https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a


        # Frame stacking is quite common in DQN
        # if we implement the frame stacking in the gym env we get easier code since sb3 and others can reuse the observations easily
        # for example for the replay buffer
        # we could also use an observation wrapper
        # frame stacking of one means there is no stacking done
        self.frame_stacking = frame_stacking

        self.grayscale = grayscale
        if grayscale:
            self.channels=1
        else:
            self.channels=3

        # we use the channel to stack the frames, let's see if that works
        if self.frame_stacking == 1:
            self.channels_total = self.channels
        else:
            self.channels_total = self.channels * self.frame_stacking
            assert self.channels_total < self.height, f'required for proper is_image_space_channels_first in sb3.common.preprocessing'
            assert self.channels_total < self.width, f'required for proper is_image_space_channels_first in sb3.common.preprocessing'
            print(f'channels total {self.channels_total}', flush=True)
        self.normalize_images = normalize_images
        if normalize_images:
            high = 1

            # sb3 does not do the normalization itself, we can do it here
            # TODO we also need to set the parameter normalized_images=True in the CnnPolicy
            # see https://github.com/DLR-RM/stable-baselines3/blob/b413f4c285bc3bfafa382559b08ce9d64a551d26/stable_baselines3/common/torch_layers.py#L48
            self.obs_dtype =  np.float32

            assert False, f'not implemented yet, did you set normalized_images=True in the CnnPolicy?'
        else:
            high = 255
            self.obs_dtype = np.uint8
            
        self.observation_space = spaces.Box(low=0,
                                    high=high,
                                    shape=(self.height, self.width, self.channels_total),
                                    dtype=self.obs_dtype)

        print(f'Observation space shape {self.observation_space.shape}', flush=True)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32)
        # TODO maybe use 0 for low instead of -1.0 (what did maximilian use?)
        # this box is essentially an array

        #print(f'Box sample {self.action_space.sample()}')
        # Box sample [[0.85317516]  [0.07102327]]

        if BaseUnityCarEnv.unity_comms is None:
            # all instances of this class share the same UnityComms instance
            # they use their self.instancenumber to differentiate between them
            # the PPCCR.cs has to be rewritten to use these instancenumbers
            BaseUnityCarEnv.unity_comms = UnityComms(port=port)
            BaseUnityCarEnv.unity_comms.deleteAllArenas()

        self.instancenumber = BaseUnityCarEnv.instancenumber

        BaseUnityCarEnv.unity_comms.startArena(
            id=self.instancenumber)
        BaseUnityCarEnv.instancenumber += 1

        # TODO check maybe we can also do self.unity_comms = BaseUnityCarEnv.unity_comms
        # this would make the rest of the code more readable

        self.spawn_point_random = spawn_point_random
        self.single_goal = single_goal

        self.mapType = MapType.random

        print(f'spawn_point_random {self.spawn_point_random} single_goal {self.single_goal}', flush=True)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        if self.asynchronous:
            return self.stepAsynchronous(action)
        else:
            return self.stepSynchronous(action)

    def stepSynchronous(self, action):
        """Perform step, return observation, reward, terminated, false, info."""

        left_acceleration, right_acceleration = action

        assert left_acceleration >= - \
            1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= - \
            1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'

        stepObj = BaseUnityCarEnv.unity_comms.immediateStep(id=self.instancenumber, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

        reward = stepObj["reward"]
        terminated = stepObj["done"]
        truncated = stepObj["done"]
        # TODO differentiate these two

        info_dict = stepObj["info"]

        # print(
        #    f'left_acceleration {left_acceleration} right_acceleration {right_acceleration}, reward {reward}')

        if terminated and False:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        new_obs = self.unityStringToObservation(stepObj["observation"])

#        print(f'max new_obs before {np.max(new_obs)} min {np.min(new_obs)}', flush=True)
        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)
 #       print(f'max new_obs after {np.max(new_obs)} min {np.min(new_obs)}', flush=True)

        return new_obs, reward, terminated, truncated, info_dict

    def stepAsynchronous(self, action):
        """Perform step, return observation, reward, terminated, false, info."""

        left_acceleration, right_acceleration = action

        assert left_acceleration >= - \
            1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= - \
            1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'

        BaseUnityCarEnv.unity_comms.asyncStepPart1(id=self.instancenumber, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

        time.sleep(0.1)
        stepObj = BaseUnityCarEnv.unity_comms.asyncStepPart2(
            id=self.instancenumber)

        reward = stepObj["reward"]
        terminated = stepObj["done"]
        truncated = stepObj["done"]
        # TODO differentiate these two

        info_dict = stepObj["info"]

        # print(
        #    f'left_acceleration {left_acceleration} right_acceleration {right_acceleration}, reward {reward}')

        if terminated and False:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)


        new_obs = self.unityStringToObservation(stepObj["observation"])
        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)

        return new_obs, reward, terminated, truncated, info_dict
        # TODO change the returned tuple to match the new gymnasium step API
        # https://www.gymlibrary.dev/content/api/#stepping
        # TODO check if there is a stable baselines 3 version ready for this new API

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)  # gynasium migration guide https://gymnasium.farama.org/content/migration-guide/

        if self.frame_stacking > 1:
            self.memory = np.zeros((self.height, self.width, self.channels_total), dtype=self.obs_dtype)

        obsstring = BaseUnityCarEnv.unity_comms.reset(mapType=self.mapType.name,
            id=self.instancenumber, spawnpointRandom=self.spawn_point_random, singleGoalTraining=self.single_goal)
        info = {}

        new_obs = self.unityStringToObservation(obsstring)

        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)

        return new_obs, info
    


    # TODO try this wrapper instead: https://github.com/DLR-RM/stable-baselines3/blob/b413f4c285bc3bfafa382559b08ce9d64a551d26/stable_baselines3/common/vec_env/vec_frame_stack.py#L12
    def memory_rollover(self, new_obs):
        # was verified with an RGB example
        # see all the commented out lines

        log = self.log

        #print(f'new obs min and max {np.min(new_obs)} {np.max(new_obs)}', flush=True)

        assert new_obs.dtype == self.obs_dtype, f'new_obs.dtype {new_obs.dtype} self.obs_dtype {self.obs_dtype}'

        channels = self.channels
        #print(f'mem shape {self.memory.shape} {self.memory.dtype} channels {channels} new_obs shape {new_obs.shape} {new_obs.dtype}', flush=True)
        
        if log:
            d = new_obs
            if self.normalize_images:
                d = d * 255.0
            if channels== 3:
                img = Image.fromarray(d, 'RGB')
                img.save(f'imagelog/new_obs.png')
            else:
                img = Image.fromarray(d, 'L')
                img.save(f'imagelog/new_obs.png')


            for i in range(self.frame_stacking):
                data = self.memory[:,:,i*channels:i*channels+channels]
                if self.normalize_images:
                    data = data * 255.0

                if channels== 3:
                    img = Image.fromarray(data, 'RGB')
                    img.save(f'imagelog/pre_rollover{i}.png')
                else:
                    #print(f'data shape {data.shape}', flush=True)
                    data = np.squeeze(data, axis=2)
                    #print(f'data shape {data.shape}', flush=True)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/pre_rollover{i}.png')

        # shift the channels to get rid of old stuff
        self.memory = np.roll(self.memory, shift=self.channels, axis=2)

        if log:
            for i in range(self.frame_stacking):
                data = self.memory[:,:,i*channels:i*channels+channels]
                if self.normalize_images:
                    data = data * 255.0
                if channels== 3:
                    img = Image.fromarray(data, 'RGB')
                    img.save(f'imagelog/post_rollover{i}.png')
                else: 
                    data = np.squeeze(data, axis=2)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/post_rollover{i}.png')

        if self.grayscale:
            new_obs = np.expand_dims(new_obs, axis=2)

        self.memory[:,:,0:self.channels] = new_obs
        
        if log:
            for i in range(self.frame_stacking):
                data = self.memory[:,:,i*channels:i*channels+channels]
                if self.normalize_images:
                    data = data * 255.0
                if channels== 3:
                    img = Image.fromarray(data, 'RGB')
                    img.save(f'imagelog/post_replace{i}.png')
                else:
                    data = np.squeeze(data, axis=2)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/post_replace{i}.png')

        return self.memory

    def getObservation(self):
        return self.unityStringToObservation(BaseUnityCarEnv.unity_comms.getObservation(id=self.instancenumber))


    def unityStringToObservation(self, obsstring):

        log = self.log

        base64_bytes = obsstring.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        #print(f'Image size {im.size}', flush=True)
        #print(f'image type {type(im)}', flush=True)
        # PIL.PngImagePlugin.PngImageFile

        pixels_rgb = np.array(im, dtype=np.uint8)
        #print(f'pixels shape {pixels.shape}', flush=True)
        # it looks like this switches the height and width
        
        #print(f'unit img max {np.max(pixels_rgb)} min {np.min(pixels_rgb)}', flush=True)

        if log:
            img = Image.fromarray(pixels_rgb, 'RGB')
            img.save("imagelog/image_from_unity.png")

        pixels_float = np.array(im, dtype=np.float32)
        #print(f'unit img float max {np.max(pixels_float)} min {np.min(pixels_float)}', flush=True)

        #print(f'pixels float shape {pixels_float.shape}', flush=True)

        if log:
            pixels_float_uint8 = pixels_float.astype(np.uint8)
            img = Image.fromarray(pixels_float_uint8, 'RGB')
            img.save("imagelog/image_from_unity_float.png")


        pixels_downsampled = block_reduce(pixels_float, block_size=(2, 2, 1), func=np.mean)
        # this halves the size along each dim


        if log:
            pixels_downsampled_uint8 = pixels_downsampled.astype(np.uint8)
            im = Image.fromarray(pixels_downsampled_uint8, 'RGB')
            im.save("imagelog/image_from_unity_downsampled.png")

        pixels_gray = color.rgb2gray(pixels_downsampled)

        #print(f'pixels gray shape {pixels_gray.shape}', flush=True)

        if log:
            pixels_gray_uint8 = pixels_gray.astype(np.uint8)
            im = Image.fromarray(pixels_gray_uint8, 'L') # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            im.save("imagelog/image_from_unity_gray.png")

        if self.equalize:
            assert self.grayscale, f'equalize only works with grayscale images'
            pixels_equalized, histOrig, histEq = hist_eq(pixels_gray)

            if log:
                pixels_equalized_uint8 = pixels_equalized.astype(np.uint8)
                im = Image.fromarray(pixels_equalized_uint8, 'L')
                im.save("imagelog/image_from_unity_equalized.png")


        if self.grayscale:
            pixels_result = pixels_gray
        else:
            pixels_result = pixels_downsampled
        
        #print(f'self normalize_images {self.normalize_images} pixels_result max {np.max(pixels_result)} min {np.min(pixels_result)}', flush=True)
        if self.normalize_images:
            
            pixels_result = pixels_result / 255.0
            #print(f'min and max after normalize_images {np.min(pixels_result)} {np.max(pixels_result)}', flush=True)
        
        
        pixels_result = pixels_result.astype(self.obs_dtype)
        #print(f'min max after dtype change {np.min(pixels_result)} {np.max(pixels_result)}', flush=True)
        
        assert pixels_result.dtype == self.obs_dtype, f'pixels_result.dtype {pixels_result.dtype} self.obs_dtype {self.obs_dtype}'

        return pixels_result

    def render(self, mode='human'):
        obs = BaseUnityCarEnv.unity_comms.getObservation(
            id=self.instancenumber)

        base64_bytes = obs.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        with open("imagepython_base64_gym_env.png", "wb") as file:
            file.write(message_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        im.save("savepath.png")


if __name__ == '__main__':

    env = BaseUnityCarEnv()
    env.reset()
    env.step((0, 0))

    check_env(env)
    print(f'now checking with stable baselines 3')
    check_env_sb3(env)
