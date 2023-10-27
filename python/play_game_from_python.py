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


def run(args: argparse.Namespace) -> None:
    print(f'will try port {args.port}', flush=True)
    unity_comms = UnityComms(port=args.port)
    print(f'Unity comms created', flush=True)

    pygame.init()

    # creating display
    gameDisplay = pygame.display.set_mode((512, 256))

    right_acceleration, left_acceleration = 0, 0

    starttime = time.time()
    frames = 0

    car_spawned = False

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

                if event.key == pygame.K_LEFT:
                    left_acceleration += 0.1
                    right_acceleration += -0.05

                if event.key == pygame.K_RIGHT:
                    right_acceleration += 0.1
                    left_acceleration += -0.05
                if event.key == pygame.K_SPACE:
                    right_acceleration = 0
                    left_acceleration = 0

                if event.key == pygame.K_r:
                    unity_comms.destroyMap()  # car has to be spawned before episode is started
                    car_spawned = False

                if event.key == pygame.K_s:
                    print("will reset the simulation")
                    unity_comms.reset()
                    car_spawned = True

                if event.key == pygame.K_p:
                    print(f'p was pressed, will pause/start the simulation')
                    unity_comms.pauseStartSimulation()
                    # this does not work on the c# side yet, the Rpc Listener essentially dies
                    # instead the timeScale should not be modified, instead we could implement the car to check if the simulation is currently paused
                    # if the sim is paused, don't turn the wheels
                    raise NotImplementedError("check comments above")

                if right_acceleration > 1:
                    right_acceleration = 1
                if right_acceleration < -0.1:
                    right_acceleration = -0.1
                if left_acceleration > 1:
                    left_acceleration = 1
                if left_acceleration < -0.1:
                    left_acceleration = -0.1

                if car_spawned:
                    stepObj = unity_comms.step(inputAccelerationLeft=float(
                        left_acceleration), inputAccelerationRight=float(right_acceleration))
                    print(
                        f'stepObj reward {stepObj["reward"]} done {stepObj["done"]} info {stepObj["info"]}', flush=True)

                if event.key == pygame.K_q or event.key == pygame.K_c:
                    print("q or c pressed, will quit", flush=True)
                    pygame.quit()
                    sys.exit()

        if car_spawned:
            obs = unity_comms.getObservation()
            with open("obs.txt", "w") as file:
                file.write(obs)

            base64_bytes = obs.encode('ascii')
            message_bytes = base64.b64decode(base64_bytes)

            with open("imagepython_base64.png", "wb") as file:
                file.write(message_bytes)

            im = Image.open(io.BytesIO(message_bytes))

            im.save("savepath.png")

            img = pygame.image.load('savepath.png')
            gameDisplay.blit(img, (0, 0))

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
