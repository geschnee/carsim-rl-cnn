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

from gymEnv.histogram_equilization import hist_eq

from gymEnv.myEnums import MapType, EndEvent, Spawn, LightSetting

import random

@dataclass
class StepReturnObject:
    previousStepNotFinished: bool
    observation: str
    done: bool
    terminated: bool
    info: dict
    rewards: list[float]

@dataclass
class StepReturnObjectList:
    objects: list[StepReturnObject]
    step_script_realtime_duration: float

class BaseCarsimEnv(gym.Env):

    unity_comms: UnityComms = None
    instancenumber = 0

    def __init__(self, width=336, height=168, port=9000, log=False, jetbot=None, spawn_point=None, fixedTimestepsLength=None, trainingMapType=MapType.randomEval, trainingLightSetting=LightSetting.random, image_preprocessing={}, frame_stacking=5, coefficients=None, collisionMode=None):
        # height and width was previous 168, that way we could downsample and reach the same dimensions as the nature paper of 84 x 84

        self.instancenumber = BaseCarsimEnv.instancenumber
        assert jetbot is not None
        self.jetbot = jetbot

        self.fixedTimestepsLength = fixedTimestepsLength

        self.read_preprocessing(image_preprocessing)
        

        self.video_filename = ""

        self.trainingLightSetting = trainingLightSetting

        self.log = log

        self.width = int(width / self.downsampling_factor)
        self.height = int(height / self.downsampling_factor)
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

        
        if self.grayscale:
            self.channels=1
        else:
            self.channels=3

        # we use the channel to stack the frames
        if self.frame_stacking == 1:
            self.channels_total = self.channels
        else:
            self.channels_total = self.channels * self.frame_stacking
            assert self.channels_total < self.height, f'required for proper is_image_space_channels_first in sb3.common.preprocessing'
            assert self.channels_total < self.width, f'required for proper is_image_space_channels_first in sb3.common.preprocessing'
            
            if self.instancenumber == 0:
                print(f'channels total {self.channels_total}', flush=True)
        
        
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
        # this box is essentially an array

        

        if BaseCarsimEnv.unity_comms is None:
            # all instances of this class share the same UnityComms instance
            # they use their self.instancenumber to differentiate between them
            BaseCarsimEnv.unity_comms = UnityComms(port=port)
            self.unityDeleteAllArenas()

        if fixedTimestepsLength:
            fixedTimesteps = True
        else:
            fixedTimesteps = False
            fixedTimestepsLength = 0
        

        self.collisionMode = collisionMode

        self.unityStartArena(width, height, jetbot, fixedTimesteps, fixedTimestepsLength)
        BaseCarsimEnv.instancenumber += 1

        self.spawn_point = spawn_point

        self.mapType = trainingMapType


    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Perform step, return observation, reward, terminated, false, info."""

        #_ = self.unityPing()

        left_acceleration, right_acceleration = action

        assert left_acceleration >= - \
            1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= - \
            1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'

        
        stepObj: StepReturnObject = self.unityImmediateStep(left_acceleration, right_acceleration)
        waitTimeStart=time.time()
        waitTime=False
        while stepObj.previousStepNotFinished:
            #print(f'waiting for previous step to finish', flush=True)
            waitTime = time.time() - waitTimeStart
            stepObj: StepReturnObject = self.unityImmediateStep(left_acceleration, right_acceleration)

        if waitTime:
            self.episodeWaitTime += waitTime

        return self.processStepReturnObject(stepObj)

    
    def bundledStep(self, step_nrs, left_actions: list[float], right_actions: list[float]) -> list[StepReturnObject]:
        
        starttime = time.time()
        stepObjList, step_script_realtime_duration = self.unityBundledStep(step_nrs, left_actions, right_actions)

        waitTimeStart=time.time()
        waitTime=False
        waiting = 0
        while not self.allPreviousStepsFinished(stepObjList):
            waitTime = time.time() - waitTimeStart
            waiting += 1
            stepObjList, step_script_realtime_duration = self.unityBundledStep(step_nrs, left_actions, right_actions)

        if waitTime:
            self.episodeWaitTime += waitTime

        #if waiting != 0:
        #    print(f'waited {waiting} times for previous step to finish, total step call duration {time.time() -starttime}', flush=True)


        # print(f'time recorded in c# step calls {step_script_realtime_duration}', flush=True) # with profiling (unityBundledStep) the last print of this can be used to determine the ratio of time between transmission time and c# processing time

        return stepObjList

    def allPreviousStepsFinished(self, stepObjList):
        assert len(stepObjList) > 0, f'stepObjList is empty'
        for stepObj in stepObjList:
            if stepObj.previousStepNotFinished:
                return False
        return True


    def processStepReturnObject(self, stepObj: StepReturnObject):
        # this function is made for the bundled calls
        
        reward = 0 # this reward is corrected in the policy later on (using the info dictionary)
        terminated = stepObj.done
        truncated = stepObj.done

        info_dict = stepObj.info
        info_dict["rewards"] = stepObj.rewards
        info_dict["episodeWaitTime"] = self.episodeWaitTime
        info_dict["spawnRot"] = self.current_spawn_rot

        self.step_nr += 1
        assert self.step_nr == int(info_dict["step"]), f'self.step {self.step_nr} info_dict["step"] {info_dict["step"]} for {self.instancenumber}'

        assert info_dict["amount_of_steps"] == info_dict["amount_of_steps_based_on_rewardlist"], f'info_dict["amount_of_steps"] {info_dict["amount_of_steps"]}, baed on rewardlist {info_dict["amount_of_steps_based_on_rewardlist"]}'


        if terminated and False:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        new_obs = self.stringToObservationStep(stepObj.observation)

        if self.frame_stacking > 1:
            new_obs = self.memory_rolloverStep(new_obs)

        return new_obs, reward, terminated, truncated, info_dict

    def unityBundledStep(self, step_nrs, left_actions, right_actions):
        objectList = BaseCarsimEnv.unity_comms.bundledStep(ResultClass=StepReturnObjectList, step_nrs=step_nrs, left_actions=left_actions, right_actions=right_actions)
        return objectList.objects, objectList.step_script_realtime_duration

    # move all calls to seperate functions for profiling
    def unityImmediateStep(self, left_acceleration, right_acceleration):
        return BaseCarsimEnv.unity_comms.immediateStep(ResultClass=StepReturnObject, id=self.instancenumber, step=self.step_nr, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

    def unityReset(self, mp_name, spawn_rot, video_filename, lightSettingName, evalMode):

        

        return BaseCarsimEnv.unity_comms.reset(mapType=mp_name,
            id=self.instancenumber, spawn_pos=self.spawn_point.name, spawn_rot=spawn_rot, lightSettingName=lightSettingName, evalMode=evalMode, video_filename=video_filename) 

    def unityGetObservation(self):
        return BaseCarsimEnv.unity_comms.getObservation(id=self.instancenumber)
    
    def unityGetObservationAllEnvs(self):
        return BaseCarsimEnv.unity_comms.getObservationAllEnvs()
    
    def unityGetObservationBytes(self):
        return BaseCarsimEnv.unity_comms.getObservationBytes(id=self.instancenumber)
    
    def unityPing(self):
        return BaseCarsimEnv.unity_comms.ping(id=self.instancenumber)

    def unityStartArena(self, width, height, jetbot, fixedTimesteps, fixedTimestepsLength):

        return BaseCarsimEnv.unity_comms.startArena(
            id=self.instancenumber, jetbotName=jetbot, distanceCoefficient=self.distanceCoefficient, orientationCoefficient=self.orientationCoefficient, velocityCoefficient=self.velocityCoefficient, eventCoefficient=self.eventCoefficient, resWidth=width, resHeight=height, fixedTimesteps=fixedTimesteps, fixedTimestepsLength=fixedTimestepsLength, collisionMode=self.collisionMode)
        
    def unityDeleteAllArenas(self):
        BaseCarsimEnv.unity_comms.deleteAllArenas()

    def unityGetArenaScreenshot(self):
        return BaseCarsimEnv.unity_comms.getArenaScreenshot(id=self.instancenumber)

    def setVideoFilename(self, video_filename):
        self.video_filename = video_filename
        #print(f'{self.instancenumber} video filename {self.video_filename}', flush=True)

    def reset(self, seed = None, mapType = None, lightSetting = None, evalMode = False, spawnRot=None):
        super().reset(seed=seed)  # gynasium migration guide https://gymnasium.farama.org/content/migration-guide/


        self.step_nr = -1
        self.step_mistakes = 0
        self.step_mistake_step = -1
        self.episodeWaitTime = 0
        
        if self.frame_stacking > 1:
            self.memory = np.zeros((self.height, self.width, self.channels_total), dtype=self.obs_dtype)

        mp_name = self.getMapTypeName(mapType=mapType)
        lightSettingName = self.getLightSettingName(lightSetting)

        spawn_rot = self.getSpawnRot(spawnRot)
        self.current_spawn_rot = spawn_rot

        obsstring = self.unityReset(mp_name, spawn_rot, video_filename=self.video_filename, lightSettingName=lightSettingName, evalMode=evalMode)

        
        info = {"mapType": mp_name, "spawnRot": self.current_spawn_rot}

        # do not take the observation from the reset, since the camera needs a frame to get sorted out
        new_obs = self.stringToObservation(self.unityGetObservation())


        if self.frame_stacking > 1:
            new_obs = self.memory_rollover(new_obs)

        return new_obs, info
    
    def resetMemory(self):
        self.memory = np.zeros((self.height, self.width, self.channels_total), dtype=self.obs_dtype)
    
    def getMapTypeName(self, mapType):
        if mapType is not None:
            mp = mapType
            assert isinstance(mp, MapType), f'mapType must be maptype, not {type(mp)}'
        else:
            mp = self.mapType
        mapTypeName = MapType.resolvePseudoEnum(mp).name
        return mapTypeName
    
    def getLightSettingName(self, lightSetting):
        if lightSetting is not None:
            ls = lightSetting
            assert isinstance(ls, LightSetting), f'lightSetting must be lightSetting enum, not {type(ls)}'
        else:
            ls = self.trainingLightSetting
        lightSettingName = LightSetting.resolvePseudoEnum(ls).name
        return lightSettingName
    
    def getSpawnRot(self, spawnRot):
        if spawnRot is None:

            assert type(self.spawn_point) == Spawn, f'spawn point must be set'

            interval_min, interval_max = Spawn.getOrientationRange(self.spawn_point)
            spawn_rot = random.randint(interval_min, interval_max)
            return spawn_rot
        else:
            assert isinstance(spawnRot, int), f'spawnRot must be int, not {type(spawnRot)}'
            return spawnRot
    
    def getSpawnMode(self):
        return self.spawn_point
    
    def reset_with_difficulty(self, difficulty, lightSetting=None, evalMode=False):
        mapType = MapType.getMapTypeFromDifficulty(difficulty)
        return self.reset(mapType=mapType, lightSetting=lightSetting, evalMode=evalMode)

    def reset_with_difficulty_spawnrotation(self, difficulty, lightSetting=None, evalMode=False, spawnRot=None):
        mapType = MapType.getMapTypeFromDifficulty(difficulty)
        return self.reset(mapType=mapType, lightSetting=lightSetting, evalMode=evalMode, spawnRot=spawnRot)
    
    def reset_with_mapType_spawnrotation(self, mapType, lightSetting=None, evalMode=False, spawnRot=None):
        return self.reset(mapType=mapType, lightSetting=lightSetting, evalMode=evalMode, spawnRot=spawnRot)


    def rollover_log_before(self, new_obs, channels):
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
                data = np.squeeze(data, axis=2)
                img = Image.fromarray(data, 'L')
                img.save(f'imagelog/{self.step_nr}_pre_rollover{i}.png')

    def rollover_log_post_rollover(self, channels):
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

    def rollover_log_post_replace(self, channels):
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

    def memory_rolloverStep(self, new_obs, log = None):
        return self.memory_rollover(new_obs, log) 

    # TODO try this wrapper instead: https://github.com/DLR-RM/stable-baselines3/blob/b413f4c285bc3bfafa382559b08ce9d64a551d26/stable_baselines3/common/vec_env/vec_frame_stack.py#L12
    def memory_rollover(self, new_obs, log = None):
        # was verified with an RGB example
        # see all the commented out lines

        if log is None:
            log = self.log

        assert new_obs.dtype == self.obs_dtype, f'new_obs.dtype {new_obs.dtype} self.obs_dtype {self.obs_dtype}'

        channels = self.channels
        
        if log:
            self.rollover_log_before(new_obs, channels)

        # shift the channels to get rid of old stuff
        self.memory = np.roll(self.memory, shift=self.channels, axis=2)

        if log:
            self.rollover_log_post_rollover(channels)

        if self.grayscale:
            new_obs = np.expand_dims(new_obs, axis=2)

        self.memory[:,:,0:self.channels] = new_obs
        
        if log:
            self.rollover_log_post_replace(channels)

        return self.memory

    def get_observation_including_memory(self, log=False):
        # this should not be used for logging some image files

        # get_observation_including_memory has 0.032 percall time
        # unityGetObservation has 0.023 percall time
        # stringToObservation has 0.009 percall time
        # for stringTOObservation the time is mostly the preprocessing

        obs_string = self.unityGetObservation()
        obs = self.stringToObservation(obs_string, log)

        #obs_bytes = self.unityGetObservationBytes()
        #obs_from_bytes = self.byteArrayToImg(obs_bytes)

        #assert np.array_equal(obs, obs_from_bytes), f'obs and obs_from_bytes are not equal'

        if self.frame_stacking > 1:
            obs = self.memory_rollover(obs, log)
        return obs
    
    def get_obsstrings_with_single_request(self, log = False):
        # TODO refactoring later on when the speed improvement is shown
        # or rather fixing (for the different envs)

        all_obsstrings = self.unityGetObservationAllEnvs()

        return all_obsstrings
        
    def setSeedUnity(self, seed):
        BaseCarsimEnv.unity_comms.setSeed(seed=seed)

    def setLog(self, log):
        self.log = log

    def getObservation(self):
        return self.stringToObservation(self.unityGetObservation())

    def saveObservationNoPreprocessing(self, filename):
        obsstring = self.unityGetObservation()
        

        im = self.stringToImg(obsstring)

        pixels_rgb = np.array(im, dtype=np.uint8)
        
        img = Image.fromarray(pixels_rgb, 'RGB')
        img.save(filename)

    def saveObservation(self, filename_prefix):
        obs = self.stringToObservation(self.unityGetObservation(), log=filename_prefix)
        im = Image.fromarray(obs, 'L') # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        im.save(f"{filename_prefix}.png")

    def saveImage(self, array, filename):
        img = Image.fromarray(array, 'RGB')
        img.save(filename)


    def saveImageGreyscale(self, array, filename):
        img = Image.fromarray(array, 'L')
        img.save(filename)

    def read_preprocessing(self, image_preprocessing):
        if image_preprocessing["downsampling_factor"]:
            self.downsampling_factor = image_preprocessing["downsampling_factor"]
            self.downsample = True
        else:
            self.downsample = False
            self.downsampling_factor = 1

            
        self.grayscale = image_preprocessing["grayscale"]

        self.equalize = image_preprocessing["equalize"]
        self.normalize_images = image_preprocessing["normalize_images"]
        if self.normalize_images:
            assert self.obs_dtype == np.float32, f'normalize_images=True requires dtype float32 (int cannot store 0-1 range, only 0-255 range)'


    def preprocessDownsample(self, pixels, log):
        x, y = self.downsampling_factor, self.downsampling_factor
        pixels_downsampled = block_reduce(pixels, block_size=(x, y, 1), func=np.mean)
        # if x==y==2 it halves the size along each dim

        if log:
            pixels_downsampled_uint8 = pixels_downsampled.astype(np.uint8)
            self.saveImage(pixels_downsampled_uint8, "imagelog/image_from_unity_downsampled.png")
            self.saveImage(pixels_downsampled_uint8, "latex_images/agent_downsampled.png")

            if type(log) == str:
                self.saveImage(pixels_downsampled_uint8, f"{log}_downsampled.png")
        
        return pixels_downsampled

    def preprocessGrayscale(self, pixels, log):

        pixels_gray = color.rgb2gray(pixels)

        if log:
            pixels_gray_uint8 = pixels_gray.astype(np.uint8)
            self.saveImageGreyscale(pixels_gray_uint8, "imagelog/image_from_unity_grey.png")
            self.saveImageGreyscale(pixels_gray_uint8, "latex_images/agent_grey.png")

            if type(log) == str:
                self.saveImageGreyscale(pixels_gray_uint8, f"{log}_greyscale.png")

        return pixels_gray
    
    def preprocessEqualize(self, pixels, log):
        assert self.grayscale, f'equalize only works with grayscale images'
        pixels_equalized, histOrig, histEq = hist_eq(pixels)

        if log:
            pixels_equalized_uint8 = pixels_equalized.astype(np.uint8)
            self.saveImageGreyscale(pixels_equalized_uint8, "imagelog/images_from_unity_equalized.png")
            self.saveImageGreyscale(pixels_equalized_uint8, "latex_images/agent_equalized.png")

            if type(log) == str:
                self.saveImageGreyscale(pixels_equalized_uint8, f"{log}_equalized.png")

        return pixels_equalized

    def preprocessNormalizeImages(self, pixels, log):
        # this just makes the pixel values in the range [0, 1]
        # this can help learn quicker

        return pixels / 255.0
    



    def stringToObservationStep(self, obsstring, log=None):
        return self.stringToObservation(obsstring, log)

    def stringToObservation(self, obsstring, log=None):

        if log is None:
            log = self.log

        im = self.stringToImg(obsstring)

        pixels_result = self.preprocessing(im, log)

        return pixels_result
    
    def preprocessing(self, im, log=None):
        preprocessing_priority = ["downsample", "grayscale", "equalize", "normalize_images"]

        pixels_rgb = np.array(im, dtype=np.uint8)
        # it looks like this switches the height and width
        
        if log:
            print("logging the image")
            if not os.path.exists('imagelog'):
                os.makedirs('imagelog')
            if not os.path.exists('latex_images'):
                os.makedirs('latex_images')

            self.saveImage(pixels_rgb, "imagelog/image_from_unity.png")
            self.saveImage(pixels_rgb, "latex_images/agent_image_from_unity.png")

            if type(log) == str:
                self.saveImage(pixels_rgb, f"{log}_image_from_unity.png")

        pixels_float = np.array(im, dtype=np.float32)
        
        
        pixels_result = pixels_float
        for step in preprocessing_priority:
            if step == "downsample":
                if self.downsample:
                    pixels_result = self.preprocessDownsample(pixels_result, log)
            elif step == "grayscale":
                if self.grayscale:
                    pixels_result = self.preprocessGrayscale(pixels_result, log)
            elif step == "equalize":
                if self.equalize:
                    pixels_result = self.preprocessEqualize(pixels_result, log)
            elif step == "normalize_images": 
                if self.normalize_images:
                    pixels_result = self.preprocessNormalizeImages(pixels_result, log)
            else:
                assert False, f'unknown step {step}'
        
        
        pixels_result = pixels_result.astype(self.obs_dtype)
        
        assert pixels_result.dtype == self.obs_dtype, f'pixels_result.dtype {pixels_result.dtype} self.obs_dtype {self.obs_dtype}'
        return pixels_result

    def render(self, mode='human'):
        obs = self.unityGetObservation()

        im = self.stringToImg(obs)

        im.save("savepath.png")
        im.save("imagepython_base64_gym_env.png")

    def stringToImg(self, string):
        base64_bytes = string.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        return im
    
    '''
    def byteArrayToImg(self, message_bytes):
        print(f'type {type(message_bytes)}', flush=True)
        print(f'length {len(message_bytes)}', flush=True)
        print(f'first 10 bytes {message_bytes[:10]}', flush=True)

        im = Image.open(io.BytesIO(message_bytes))

        return im'''

    def get_arena_screenshot(self, savepath="arena_screenshot.png"):
        screenshot = self.unityGetArenaScreenshot()

        im = self.stringToImg(screenshot)

        im.save(savepath)
        return im

if __name__ == '__main__':

    env = BaseCarsimEnv()
    env.reset()
    env.step((0, 0))

    check_env(env)
    print(f'now checking with stable baselines 3')
    check_env_sb3(env)
