from typing import Any, SupportsFloat
import gymnasium.spaces as spaces
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from stable_baselines3.common.env_checker import check_env as check_env_sb3

import numpy as np

from peaceful_pie.unity_comms import UnityComms


import PIL.Image as Image
import io
import base64

from dataclasses import dataclass

import time


@dataclass
class StepReturnObject:
    obs: str
    reward: float
    done: bool
    terminated: bool
    info: dict


class BaseUnityCarEnv(gym.Env):

    unity_comms: UnityComms = None
    instancenumber = 0

    def __init__(self, width=512, height=256, port=9000, asyncronous=True):

        self.asynchronous = asyncronous
        self.width = width
        self.height = height
        # width and height in pixels of the screen

        # how are RL algos trained for continuous action spaces?
        # https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.height, self.width, 3),
                                            dtype=np.uint8)
        # shape is height, width, 3 since we use three channels for the colors
        # the height and width are in that order since the np.array switches them, see unityStringToObservation
        # sb3 PPO CnnPolicy requires high being 255 and dtype uint8

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32)
        # TODO maybe use 0 for low instead of -1.0 (what did maximilian use?)
        # this box is essentially an array

        print(f'Box sample {self.action_space.sample()}')
        # Box sample [[0.85317516]  [0.07102327]]

        if BaseUnityCarEnv.unity_comms is None:
            # all instances of this class share the same UnityComms instance
            # they use their self.instancenumber to differentiate between them
            # the PPCCR.cs has to be rewritten to use these instancenumbers
            BaseUnityCarEnv.unity_comms = UnityComms(port=port)

        self.instancenumber = BaseUnityCarEnv.instancenumber
        BaseUnityCarEnv.unity_comms.startArena(self.instancenumber)
        BaseUnityCarEnv.instancenumber += 1

        # TODO check maybe we can also do self.unity_comms = BaseUnityCarEnv.unity_comms
        # this would make the rest of the code more readable

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

        stepObj = BaseUnityCarEnv.unity_comms.immediateStep(self.instancenumber, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

        reward = stepObj["reward"]
        terminated = stepObj["done"]
        truncated = stepObj["done"]
        # TODO differentiate these two

        info_dict = stepObj["info"]

        print(
            f'left_acceleration {left_acceleration} right_acceleration {right_acceleration}, reward {reward}')

        if terminated:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        return self.unityStringToObservation(stepObj["observation"]), reward, terminated, truncated, info_dict

    def stepAsynchronous(self, action):
        """Perform step, return observation, reward, terminated, false, info."""

        left_acceleration, right_acceleration = action

        assert left_acceleration >= - \
            1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= - \
            1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'

        BaseUnityCarEnv.unity_comms.asyncStepPart1(self.instancenumber, inputAccelerationLeft=float(
            left_acceleration), inputAccelerationRight=float(right_acceleration))

        time.sleep(0.1)
        stepObj = BaseUnityCarEnv.unity_comms.asyncStepPart2(
            self.instancenumber)

        reward = stepObj["reward"]
        terminated = stepObj["done"]
        truncated = stepObj["done"]
        # TODO differentiate these two

        info_dict = stepObj["info"]

        print(
            f'left_acceleration {left_acceleration} right_acceleration {right_acceleration}, reward {reward}')

        if terminated:
            print(
                f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        return self.unityStringToObservation(stepObj["observation"]), reward, terminated, truncated, info_dict
        # TODO change the returned tuple to match the new gymnasium step API
        # https://www.gymlibrary.dev/content/api/#stepping
        # TODO check if there is a stable baselines 3 version ready for this new API

    def reset(self, seed=None, **kwargs):
        """Place 2 tiles on empty board."""
        super().reset(seed=seed)  # gynasium migration guide https://gymnasium.farama.org/content/migration-guide/

        obsstring = BaseUnityCarEnv.unity_comms.reset(self.instancenumber)
        info = {}

        return self.unityStringToObservation(obsstring), info

    def getObservation(self):
        return self.unityStringToObservation(BaseUnityCarEnv.unity_comms.getObservation(self.instancenumber))

    def unityStringToObservation(self, obsstring):
        base64_bytes = obsstring.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        #print(f'Image size {im.size}', flush=True)
        #print(f'image type {type(im)}', flush=True)
        # PIL.PngImagePlugin.PngImageFile

        pixels = np.array(im, dtype=np.uint8)
        #print(f'pixels shape {pixels.shape}', flush=True)
        # it looks like this switches the height and width

        # pixels = pixels / 255.0

        return pixels

    def render(self, mode='human'):
        obs = BaseUnityCarEnv.unity_comms.getObservation(self.instancenumber)

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
