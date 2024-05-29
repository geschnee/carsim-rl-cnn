import time

import pygame
import sys

import hydra
from omegaconf import OmegaConf

import os

from gymEnv.carsimGymEnv import BaseCarsimEnv

import numpy as np

from gymEnv.myEnums import MapType
from gymEnv.myEnums import LightSetting
from gymEnv.myEnums import SpawnOrientation


def gray(im):
    w, h = im.shape[0], im.shape[1]
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def run(cfg) -> None:
    port = 9000
    print(f'will try port {port}', flush=True)

    
  
    pygame.init()

    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    my_font = pygame.font.SysFont('Comic Sans MS', 30)

    # creating display
    gameDisplay = pygame.display.set_mode((1024, 512))

    right_acceleration, left_acceleration = 0, 0

    starttime = time.time()
    frames = 0

    env_kwargs = OmegaConf.to_container(cfg.env_kwargs)
    env_kwargs["trainingMapType"] = MapType[cfg.env_kwargs.trainingMapType]
    env_kwargs["trainingLightSetting"] = LightSetting[cfg.env_kwargs.trainingLightSetting]
    env_kwargs["spawnOrientation"] = SpawnOrientation[cfg.env_kwargs.spawnOrientation]



    env = BaseCarsimEnv(**env_kwargs)

    
    env.setVideoFilename(f"{os.getcwd()}/play")
    new_obs, info_dict = env.reset()


    print(f'event reset')

    full_control_mode = False

    # creating a running loop
    while True:

        # creating a loop to check events that
        # are occurring
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    right_acceleration += 0.1
                    left_acceleration += 0.1

                if event.key == pygame.K_DOWN:
                    right_acceleration += -0.1
                    left_acceleration += -0.1

                if event.key == pygame.K_RIGHT:
                    left_acceleration += 0.1
                    if right_acceleration > 0.05:
                        right_acceleration -= 0.05

                if event.key == pygame.K_LEFT:
                    right_acceleration += 0.1
                    if left_acceleration > 0.05:
                        left_acceleration -= 0.05

                # precise/absolute control
                if event.key == pygame.K_0:
                    right_acceleration += 0.1
                    full_control_mode = True
                if event.key == pygame.K_9:
                    right_acceleration -= 0.1
                    full_control_mode = True
                if event.key == pygame.K_1:
                    left_acceleration += 0.1
                    full_control_mode = True
                if event.key == pygame.K_2:
                    left_acceleration -= 0.1
                    full_control_mode = True

                if event.key == pygame.K_SPACE:
                    right_acceleration = 0
                    left_acceleration = 0

                if event.key == pygame.K_r:
                    env.reset()

                if right_acceleration > 1:
                    right_acceleration = 1
                if left_acceleration > 1:
                    left_acceleration = 1
                if right_acceleration < -1:
                    right_acceleration = -1
                if left_acceleration < -1:
                    left_acceleration = -1
                
                if not full_control_mode:
                    if right_acceleration < -0.5:
                        right_acceleration = -0.5
                    if left_acceleration < -0.5:
                        left_acceleration = -0.5
                

                print(f'left_acceleration {left_acceleration} right_acceleration {right_acceleration}', flush=True)

                new_obs, reward, terminated, truncated, info_dict  = env.step((float(
                    left_acceleration), float(right_acceleration)))
                
                distance_reward = float(info_dict["distanceReward"].replace(",","."))
                velocity_reward = float(info_dict["velocityReward"].replace(",","."))
                event_reward = float(info_dict["eventReward"].replace(",","."))
                orientation_reward = float(info_dict["orientationReward"].replace(",","."))

                print(f'distance_reward {distance_reward} velocity_reward {velocity_reward} event_reward {event_reward} orientation_reward {orientation_reward}', flush=True)

                if terminated:
                    print(f'stepObj reward {reward} terminated {terminated} info {info_dict}', flush=True)
                    print(f'endStatus {info_dict["endEvent"]}', flush=True)
                if event.key == pygame.K_q or event.key == pygame.K_c or terminated:
                    print("episode terminated or q or c pressed, will quit", flush=True)
                    pygame.quit()
                    sys.exit()

        
            #print(f'new_obs {new_obs.shape}', flush=True)
            #print(f'max and min of new_obs {np.max(new_obs)} {np.min(new_obs)}', flush=True)

            text_surface = my_font.render(f'Input Left: {left_acceleration}\nInput right: {right_acceleration}', False, (0, 0, 0))
            gameDisplay.blit(text_surface, (600,0))

            
            if env.grayscale:
                if len(new_obs.shape) == 3:
                    img_obs = new_obs[:, :, 0:1] # get first frame
                else:
                    img_obs = new_obs
                
                img_obs = np.squeeze(img_obs)

                img_obs = np.rot90(img_obs, k=1)
                #print(f'max and min of img_obs {np.max(img_obs)} {np.min(img_obs)}', flush=True)
                g = gray(img_obs)
                #print(f'max and min of g {np.max(g)} {np.min(g)}', flush=True)

                g = np.flipud(g)
                img = pygame.surfarray.make_surface(g)
            else:
                img = new_obs[:, :, 0:3]
                img = np.rot90(img, k=1)

            
            #print(f'max and min of img {np.max(img)} {np.min(img)}', flush=True)
            
            

            gameDisplay.blit(img, (0, 0))

            arenaImg = env.get_arena_screenshot()
            arenaImg = np.rot90(arenaImg, k=1)
            arenaImg = np.flipud(arenaImg)
            arenaImg = pygame.surfarray.make_surface(arenaImg)
            gameDisplay.blit(arenaImg, (256, 0))

            pygame.display.update()

            frames += 1

            if frames % 1000 == 0:
                print(f'fps {frames / (time.time() - starttime)}')
                # about 20 fps on my machine

            #print(f'format {im.format}, size {im.size}, mode {im.mode}')
            # png, (240,240), RGB


@hydra.main(config_path=".", config_name="cfg/play_game.yaml")
def main(cfg):

    run(cfg.cfg)

if __name__ == "__main__":
    main()


