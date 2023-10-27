
import random
import gym.spaces as spaces
import gym
import numpy as np
import argparse

from peaceful_pie.unity_comms import UnityComms

import time

import pygame
import sys


import PIL.Image as Image
import io
import base64

from dataclasses import dataclass


@dataclass
class StepReturnObject:
    obs: str
    reward: float
    done: bool
    terminated: bool
    info: dict
                    
class BaseUnityCarEnv(gym.Env):
    def __init__(self, width=512, height=256, port = 9000):
        self.width = width
        self.height = height
        # width and height in pixels of the screen

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.width, self.height, 3),
                                            dtype=np.int64)
        # shape is width, height, 3 since we use three channels for the colors

        self.action_space = spaces.Discrete(4)
        # TODO find out how to represent our action

        self.unity_comms = UnityComms(port=port)

    def step(self, left_acceleration: float, right_acceleration: float):
        """Perform step, return observation, reward, terminated, false, info."""

        assert left_acceleration >= -1 and left_acceleration <= 1, f'left_acceleration {left_acceleration} is not in range [-1, 1]'
        assert right_acceleration >= -1 and right_acceleration <= 1, f'right_acceleration {right_acceleration} is not in range [-1, 1]'


        stepObj = self.unity_comms.step(inputAccelerationLeft=float(
                        left_acceleration), inputAccelerationRight=float(right_acceleration))
        reward = stepObj["reward"]
        terminated = stepObj["done"]
        info_dict = stepObj["info"]
        print(
            f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

        # TODO turn stringinto observation

        return obs, reward, terminated, info_dict
        # TODO change the returned tuple to match the new gymnasium step API
        # https://www.gymlibrary.dev/content/api/#stepping
        # it should then return this:
        # return self.board, reward, terminated, False, {"max_block" : np.max(self.board), "end_value": np.sum(self.board), "is_success": np.max(self.board) >= 2048}
        # stable-baselines3 is not ready for this change yet

    def reset(self, seed=None, **kwargs):
        """Place 2 tiles on empty board."""
        # super().reset(seed=seed) # gynasium migration guide https://gymnasium.farama.org/content/migration-guide/

        return self.board
    
    def render(self, mode='human'):
        obs = self.unity_comms.getObservation()
            

        base64_bytes = obs.encode('ascii')
        message_bytes = base64.b64decode(base64_bytes)

        with open("imagepython_base64_gym_env.png", "wb") as file:
            file.write(message_bytes)

        im = Image.open(io.BytesIO(message_bytes))

        im.save("savepath.png")

    