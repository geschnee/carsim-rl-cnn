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

import os

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
    rewards: list[float]



from enum import Enum
class Spawn(Enum):
    Fixed = 0
    OrientationRandom = 1
    FullyRandom = 2


class MapType(Enum):
    random = 0
    easyGoalLaneMiddleBlueFirst = 1
    easyGoalLaneMiddleRedFirst = 2

    twoGoalLanesBlueFirstLeftMedium = 3
    twoGoalLanesBlueFirstRightMedium = 4
    twoGoalLanesRedFirstLeftMedium = 5
    twoGoalLanesRedFirstRightMedium = 6

    twoGoalLanesBlueFirstLeftHard = 7
    twoGoalLanesBlueFirstRightHard = 8
    twoGoalLanesRedFirstLeftHard = 9
    twoGoalLanesRedFirstRightHard = 10

    # pseudo enums types
    randomEvalEasy = 11 
    randomEvalMedium = 12
    randomEvalHard = 13
    randomEval=14

    @classmethod
    def resolvePseudoEnum(myEnum, pseudoEnum):
        if pseudoEnum.value == 0:
            return myEnum.getRandomEval()
        elif pseudoEnum.value == 11:
            return myEnum.getRandomEasy()
        elif pseudoEnum.value == 12:
            return myEnum.getRandomMedium()
        elif pseudoEnum.value == 13:
            return myEnum.getRandomHard()
        elif pseudoEnum.value == 14:
            return myEnum.getRandomEval()
        else:
            # pseudoEnum is not a pseudo enum (real enum)
            return pseudoEnum

    @classmethod
    def getRandomEasy(myEnum):
        return MapType(np.random.choice([1,2]))
    
    @classmethod
    def getRandomMedium(myEnum):
        return MapType(np.random.choice([3,4,5,6]))
    
    @classmethod
    def getRandomHard(myEnum):
        return MapType(np.random.choice([7,8,9,10]))
    
    @classmethod
    def getMapTypeFromDifficulty(myEnum, difficulty):
        if difficulty == "easy":
            return myEnum.getRandomEasy()
        elif difficulty == "medium":
            return myEnum.getRandomMedium()
        elif difficulty == "hard":
            return myEnum.getRandomHard()
        else:
            assert False, f'unknown difficulty {difficulty}'

    @classmethod
    def getRandomEval(myEnum):
        return MapType(np.random.choice([1,2,3,4,5,6,7,8,9,10]))

class EndEvent(Enum):
    NotEnded = 0
    Success = 1
    OutOfTime = 2
    WallHit = 3
    GoalMissed = 4
    RedObstacle = 5
    BlueObstacle = 6

class BaseCarsimEnv(gym.Env):

    unity_comms: UnityComms = None
    instancenumber = 0

    def __init__(self, width=336, height=168, port=9000, log=False, spawn_point=None, trainingMapType=MapType.randomEval, single_goal=False, image_preprocessing={}, frame_stacking=5, lighting=1, coefficients=None):
        # height and width was previous 168, that way we could downsample and reach the same dimensions as the nature paper of 84 x 84
        self.instancenumber = BaseCarsimEnv.instancenumber

        self.equalize = image_preprocessing["equalize"]
        self.downsampling = 2

        self.lighting = lighting

        self.log = log

        self.width = int(width / self.downsampling)
        self.height = int(height / self.downsampling)
        # width and height in pixels of the screen

        # how are RL algos trained for continuous action spaces?
        # https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a

        assert coefficients is not None, f'coefficients must be set'
        self.distanceCoefficient = coefficients["distanceCoefficient"]
        self.orientationCoefficient = coefficients["orientationCoefficient"]
        self.velocityCoefficient = coefficients["velocityCoefficient"]
        self.eventCoefficient = coefficients["eventCoefficient"]

        # Frame stacking is quite common in DQN
        # if we implement the frame stacking in the gym env we get easier code since sb3 and others can reuse the observations easily
        # for example for the replay buffer
        # we could also use an observation wrapper
        # frame stacking of one means there is no stacking done
        self.frame_stacking = frame_stacking

        self.grayscale = image_preprocessing["grayscale"]
        
        if self.grayscale:
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
            
            if self.instancenumber == 0:
                print(f'channels total {self.channels_total}', flush=True)
        self.normalize_images = image_preprocessing["normalize_images"]
        
        if self.normalize_images:
            high = 1

            # sb3 does not do the normalization itself, we can do it here
            # TODO we also need to set the parameter normalize_images=True in the CnnPolicy
            # see https://github.com/DLR-RM/stable-baselines3/blob/b413f4c285bc3bfafa382559b08ce9d64a551d26/stable_baselines3/common/torch_layers.py#L48
            self.obs_dtype =  np.float32

            assert False, f'not implemented yet, did you set normalize_images=True in the CnnPolicy?'
        else:
            high = 255
            self.obs_dtype = np.uint8
            
        self.observation_space = spaces.Box(low=0,
                                    high=high,
                                    shape=(self.height, self.width, self.channels_total),
                                    dtype=self.obs_dtype)

        if self.instancenumber == 0:
            print(f'Observation space shape {self.observation_space.shape}', flush=True)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32)
        # TODO maybe use 0 for low instead of -1.0 (what did maximilian use?)
        # this box is essentially an array

        

        if BaseCarsimEnv.unity_comms is None:
            # all instances of this class share the same UnityComms instance
            # they use their self.instancenumber to differentiate between them
            # the PPCCR.cs has to be rewritten to use these instancenumbers
            BaseCarsimEnv.unity_comms = UnityComms(port=port)
            self.unityDeleteAllArenas()

        self.unityStartArena(width, height)
        BaseCarsimEnv.instancenumber += 1

        self.spawn_point = spawn_point
        self.single_goal = single_goal

        self.mapType = trainingMapType

        if self.instancenumber == 0:
            print(f'spawn_point {self.spawn_point} single_goal {self.single_goal}', flush=True)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        
        return self.stepSynchronous(action)

    def stepSynchronous(self, action):
        """Perform step, return observation, reward, terminated, false, info."""

        left_acceleration, right_acceleration = action

        assert left_acceleration >= - \
            1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= - \
            1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'

        #print(f'{self.instancenumber} step {self.step_nr} left {left_acceleration} right {right_acceleration}')
        stepObj = self.unityImmediateStep(left_acceleration, right_acceleration)

        reward = stepObj["reward"]
        terminated = stepObj["done"]
        truncated = stepObj["done"]

        info_dict = stepObj["info"]
        info_dict["rewards"] = stepObj["rewards"]

        self.step_nr += 1
        assert self.step_nr == int(info_dict["step"]), f'self.step {self.step_nr} info_dict["step"] {info_dict["step"]} for {self.instancenumber}'

        assert info_dict["amount_of_steps"] == info_dict["amount_of_steps_based_on_rewardlist"], f'info_dict["amount_of_steps"] {info_dict["amount_of_steps"]}, baed on rewardlist {info_dict["amount_of_steps_based_on_rewardlist"]}'


        if terminated and False:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        new_obs = self.stringToObservation(stepObj["observation"])

        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)

        return new_obs, reward, terminated, truncated, info_dict

    # move all calls to seperate functions for profiling
    def unityImmediateStep(self, left_acceleration, right_acceleration):
        return BaseCarsimEnv.unity_comms.immediateStep(id=self.instancenumber, step=self.step_nr, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

    def unityReset(self, mp_name):
        return BaseCarsimEnv.unity_comms.reset(mapType=mp_name,
            id=self.instancenumber, spawn=self.spawn_point, singleGoalTraining=self.single_goal, lightMultiplier = self.lighting) 

    def unityGetObservation(self):
        return BaseCarsimEnv.unity_comms.getObservation(id=self.instancenumber)
    
    def unityStartArena(self, width, height):

        return BaseCarsimEnv.unity_comms.startArena(
            id=self.instancenumber, distanceCoefficient=self.distanceCoefficient, orientationCoefficient=self.orientationCoefficient, velocityCoefficient=self.velocityCoefficient, eventCoefficient=self.eventCoefficient, resWidth=width, resHeight=height )
        
    def unityDeleteAllArenas(self):
        BaseCarsimEnv.unity_comms.deleteAllArenas()

    def unityGetArenaScreenshot(self):
        return BaseCarsimEnv.unity_comms.self.unityGetArenaScreenshot(id=self.instancenumber)


    def reset(self, seed=None, mapType = None):
        super().reset(seed=seed)  # gynasium migration guide https://gymnasium.farama.org/content/migration-guide/

        self.step_nr = -1
        self.step_mistakes = 0
        self.step_mistake_step = -1

        
        if self.frame_stacking > 1:
            self.memory = np.zeros((self.height, self.width, self.channels_total), dtype=self.obs_dtype)

        mp_name = self.getMapTypeName(mapType=mapType)

        obsstring = self.unityReset(mp_name) 
        # TODO lighting lighting=self.lighting)

        
        info = {"mapType": mp_name}

        # do not take the observation from the reset, since the camera needs a frame to get sorted out
        # between the two jrpc calls this should happen
        new_obs = self.stringToObservation(self.unityGetObservation())                                               
        # obsstring)


        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)

        return new_obs, info
    
    def getMapTypeName(self, mapType):
        if mapType is not None:
            mp = mapType
            assert not isinstance(mp, str), f'mapType must be maptype, not {type(mp)}'
        else:
            mp = self.mapType
        mapTypeName = MapType.resolvePseudoEnum(mp).name
        return mapTypeName
    
    def setRandomEval(self, randomEval):
        self.randomEval = randomEval
    
    def reset_with_difficulty(self, difficulty):
        mapType = MapType.getMapTypeFromDifficulty(difficulty)
        return self.reset(mapType=mapType)


    # TODO try this wrapper instead: https://github.com/DLR-RM/stable-baselines3/blob/b413f4c285bc3bfafa382559b08ce9d64a551d26/stable_baselines3/common/vec_env/vec_frame_stack.py#L12
    def memory_rollover(self, new_obs, log = None):
        # was verified with an RGB example
        # see all the commented out lines

        if log is None:
            log = self.log

        #print(f'new obs min and max {np.min(new_obs)} {np.max(new_obs)}', flush=True)

        assert new_obs.dtype == self.obs_dtype, f'new_obs.dtype {new_obs.dtype} self.obs_dtype {self.obs_dtype}'

        channels = self.channels
        #print(f'mem shape {self.memory.shape} {self.memory.dtype} channels {channels} new_obs shape {new_obs.shape} {new_obs.dtype}', flush=True)
        
        if log:
            
            if not os.path.exists('imagelog'):
                os.makedirs('imagelog')

            d = new_obs
            if self.normalize_images:
                d = d * 255.0
            if channels== 3:
                img = Image.fromarray(d, 'RGB')
                img.save(f'imagelog/{self.step_nr}_new_obs.png')
            else:
                img = Image.fromarray(d, 'L')
                img.save(f'imagelog/{self.step_nr}_new_obs.png')


            for i in range(self.frame_stacking):
                data = self.memory[:,:,i*channels:i*channels+channels]
                if self.normalize_images:
                    data = data * 255.0

                if channels== 3:
                    img = Image.fromarray(data, 'RGB')
                    img.save(f'imagelog/{self.step_nr}_pre_rollover{i}.png')
                else:
                    #print(f'data shape {data.shape}', flush=True)
                    data = np.squeeze(data, axis=2)
                    #print(f'data shape {data.shape}', flush=True)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/{self.step_nr}_pre_rollover{i}.png')

        # shift the channels to get rid of old stuff
        self.memory = np.roll(self.memory, shift=self.channels, axis=2)

        if log:
            for i in range(self.frame_stacking):
                data = self.memory[:,:,i*channels:i*channels+channels]
                if self.normalize_images:
                    data = data * 255.0
                if channels== 3:
                    img = Image.fromarray(data, 'RGB')
                    img.save(f'imagelog/{self.step_nr}_post_rollover{i}.png')
                else: 
                    data = np.squeeze(data, axis=2)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/{self.step_nr}_post_rollover{i}.png')

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
                    img.save(f'imagelog/{self.step_nr}_post_replace{i}.png')
                else:
                    data = np.squeeze(data, axis=2)
                    img = Image.fromarray(data, 'L')
                    img.save(f'imagelog/{self.step_nr}_post_replace{i}.png')

        return self.memory

    def get_observation_including_memory(self, log=False):
        # this should not be used for logging some image files

        obs_string = self.unityGetObservation()
        obs = self.stringToObservation(obs_string, log)

        if self.frame_stacking > 1:
            obs = self.memory_rollover(obs, log)
        return obs
    
    def setLog(self, log):
        self.log = log

    def getObservation(self):
        return self.stringToObservation(self.unityGetObservation())

    def saveObservationNoPreprocessing(self, filename):
        obsstring = self.unityGetObservation()
        base64_bytes = obsstring.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        pixels_rgb = np.array(im, dtype=np.uint8)
        
        img = Image.fromarray(pixels_rgb, 'RGB')
        img.save(filename)

    def stringToObservation(self, obsstring, log=None):
        if log is None:
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

            if not os.path.exists('imagelog'):
                os.makedirs('imagelog')
            if not os.path.exists('expose_images'):
                os.makedirs('expose_images')

            img = Image.fromarray(pixels_rgb, 'RGB')
            img.save("imagelog/image_from_unity.png")
            img.save("expose_images/agent_image_from_unity.png")

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
            im.save("expose_images/agent_downsampled.png")

        pixels_gray = color.rgb2gray(pixels_downsampled)

        #print(f'pixels gray shape {pixels_gray.shape}', flush=True)

        if log:
            pixels_gray_uint8 = pixels_gray.astype(np.uint8)
            im = Image.fromarray(pixels_gray_uint8, 'L') # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            im.save("imagelog/image_from_unity_grey.png")
            im.save("expose_images/agent_grey.png")

        if self.equalize:
            assert self.grayscale, f'equalize only works with grayscale images'
            pixels_equalized, histOrig, histEq = hist_eq(pixels_gray)

            if log:
                pixels_equalized_uint8 = pixels_equalized.astype(np.uint8)
                im = Image.fromarray(pixels_equalized_uint8, 'L')
                im.save("imagelog/image_from_unity_equalized.png")
                im.save("expose_images/agent_equalized.png")

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
        obs = self.unityGetObservation()

        base64_bytes = obs.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        with open("imagepython_base64_gym_env.png", "wb") as file:
            file.write(message_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        im.save("savepath.png")


    def get_arena_screenshot(self, savepath="arena_screenshot.png"):
        screenshot = self.unityGetArenaScreenshot()
        base64_bytes = screenshot.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        im.save(savepath)
        return im

if __name__ == '__main__':

    env = BaseCarsimEnv()
    env.reset()
    env.step((0, 0))

    check_env(env)
    print(f'now checking with stable baselines 3')
    check_env_sb3(env)
