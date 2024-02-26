import argparse

from peaceful_pie.unity_comms import UnityComms

import time

import pygame
import sys

import gymEnv.carsimGymEnv as carsimGymEnv


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import PIL.Image as Image
import io
import base64

from dataclasses import dataclass

from gymEnv.carsimGymEnv import BaseCarsimEnv

import numpy as np

from gymEnv.myEnums import MapType
from gymEnv.myEnums import Spawn

import os

def gray(im):
    w, h = im.shape[0], im.shape[1]
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def run(cfg) -> None:
    port = 9000
    print(f'will try port {port}', flush=True)

    # creating a running loop
    spawntypes = [Spawn.Fixed, Spawn.OrientationRandom, Spawn.OrientationVeryRandom, Spawn.FullyRandom]
    
    mps = [MapType(i) for i in range(11)]

    lighting_settings = {"low": {"low": 2.5, "high": 2.5}, "standard": {"low": 5.0, "high": 5.0}, "high": {"low": 7.5, "high": 7.5}}


    directory = "resetlog"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for spawn in spawntypes:
        spawn = spawn.name
        spwandir = os.path.join(directory, spawn)
        if not os.path.exists(spwandir):
            os.makedirs(spwandir)
        
        for mapType in mps:
            mapdir = os.path.join(spwandir, str(mapType))
            if not os.path.exists(mapdir):
                os.makedirs(mapdir)


            for lighting_name, lighting_setting in lighting_settings.items():

                env_kwargs = cfg.env_kwargs
                env_kwargs["trainingMapType"] = mapType
                env_kwargs["spawn_point"] = spawn
                env_kwargs["lighting_setting"] = lighting_setting

                lightdir = os.path.join(mapdir, lighting_name)
                if not os.path.exists(lightdir):
                    os.makedirs(lightdir)

                print(f"running {spawn} {mapType} {lighting_name}", flush=True)

                env = BaseCarsimEnv(**env_kwargs)
                for i in range(20):
                    #filepath = os.path.join(mapdir, f"{i}_obs.png")
                    savepath = os.path.join(lightdir, f"{i}_arena.png")
                    prefix = os.path.join(lightdir, f"{i}_")
                    
                    new_obs, info_dict = env.reset()
                    time.sleep(0.1)

                    #env.saveObservationNoPreprocessing(filepath)
                    env.saveObservation(filename_prefix=prefix)
                    env.get_arena_screenshot(savepath)

    print("check outputs/... for the results")
        


@hydra.main(config_path=".", config_name="cfg/ppo.yaml")
def main(cfg):

    run(cfg.cfg)

if __name__ == "__main__":
    main()


