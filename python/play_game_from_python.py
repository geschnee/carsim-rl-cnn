import argparse

from peaceful_pie.unity_comms import UnityComms

import time

import pygame
import sys


import PIL.Image as Image
import io
import base64

from dataclasses import dataclass

from unityGymEnv import BaseUnityCarEnv

import numpy as np

@dataclass
class StepReturnObject:
    obs: str
    reward: float
    done: bool
    terminated: bool
    info: dict


def gray(im):
    w, h = im.shape[0], im.shape[1]
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def run(args: argparse.Namespace) -> None:
    print(f'will try port {args.port}', flush=True)
    unity_comms = UnityComms(port=args.port)
    print(f'Unity comms created', flush=True)

    pygame.init()

    # creating display
    gameDisplay = pygame.display.set_mode((1024, 512))

    right_acceleration, left_acceleration = 0, 0

    starttime = time.time()
    frames = 0

    car_spawned = False

    env = BaseUnityCarEnv(frame_stacking=2, asynchronous=False, grayscale=True)
    new_obs, info_dict = env.reset()

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
                    right_acceleration += -0.05

                if event.key == pygame.K_LEFT:
                    right_acceleration += 0.1
                    left_acceleration += -0.05
                if event.key == pygame.K_SPACE:
                    right_acceleration = 0
                    left_acceleration = 0

                if event.key == pygame.K_r:
                    env.reset()

                if right_acceleration > 1:
                    right_acceleration = 1
                if right_acceleration < -0.1:
                    right_acceleration = -0.1
                if left_acceleration > 1:
                    left_acceleration = 1
                if left_acceleration < -0.1:
                    left_acceleration = -0.1

                
                new_obs, reward, terminated, truncated, info_dict  = env.step((float(
                    left_acceleration), float(right_acceleration)))
                print(
                    f'stepObj reward {reward} terminated {terminated} info {info_dict}', flush=True)

                if event.key == pygame.K_q or event.key == pygame.K_c:
                    print("q or c pressed, will quit", flush=True)
                    pygame.quit()
                    sys.exit()

        
            #print(f'new_obs {new_obs.shape}', flush=True)
            #print(f'max and min of new_obs {np.max(new_obs)} {np.min(new_obs)}', flush=True)

            

            print(f'new_obs {new_obs.shape}', flush=True)
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
                img = pygame.surfarray.make_surface(g)
            else:
                img = new_obs[:, :, 0:3]
                img = np.rot90(img, k=1)

            
            #print(f'max and min of img {np.max(img)} {np.min(img)}', flush=True)
            
            gameDisplay.blit(img, (0, 0))

            arenaImg = env.get_arena_screenshot()
            arenaImg = np.rot90(arenaImg, k=1)
            arenaImg = pygame.surfarray.make_surface(arenaImg)
            gameDisplay.blit(arenaImg, (256, 0))

            pygame.display.update()

            frames += 1

            if frames % 1000 == 0:
                print(f'fps {frames / (time.time() - starttime)}')
                # about 20 fps on my machine

            #print(f'format {im.format}, size {im.size}, mode {im.mode}')
            # png, (240,240), RGB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
