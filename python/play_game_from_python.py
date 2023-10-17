import argparse

from peaceful_pie.unity_comms import UnityComms

import time

import pygame
import sys


def run(args: argparse.Namespace) -> None:
    print(f'will try port {args.port}', flush=True)
    unity_comms = UnityComms(port=args.port)
    print(f'Unity comms created', flush=True)
    t = time.time()
    res = unity_comms.getHeight()
    print(f'Request took {time.time() - t} seconds')
    print("res", res, flush=True)

    pygame.init()

    # creating display
    display = pygame.display.set_mode((500, 500))

    right_acceleration, left_acceleration = 0, 0

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

                # if keydown event happened
                # than printing a string to output
                print("A key has been pressed", flush=True)

                if event.key == pygame.K_UP:
                    print("Key arrow up has been pressed", flush=True)
                    right_acceleration += 1
                    left_acceleration += 1

                if event.key == pygame.K_DOWN:
                    print("Key arrow down has been pressed", flush=True)
                    right_acceleration += -1
                    left_acceleration += -1

                if event.key == pygame.K_LEFT:
                    print("Key arrow left has been pressed", flush=True)
                    left_acceleration += 1

                if event.key == pygame.K_RIGHT:
                    print("Key arrow right has been pressed", flush=True)
                    right_acceleration += 1
                if event.key == pygame.K_SPACE:
                    print("Key spacebar has been pressed", flush=True)
                    right_acceleration = 0
                    left_acceleration = 0

                if event.key == pygame.K_p:
                    print(f'p was pressed, will pause/start the simulation')
                    unity_comms.pauseStartSimulation()
                    # this does not work on the c# side yet, the Rpc Listener essentially dies
                    # instead the timeScale should not be modified, instead we could implement the car to check if the simulation is currently paused
                    # if the sim is paused, don't turn the wheels
                    raise NotImplementedError("check comments above")

                unity_comms.forwardInputsToCar(inputAccelerationLeft=float(
                    left_acceleration), inputAccelerationRight=float(right_acceleration))

                if event.key == pygame.K_q or event.key == pygame.K_c:
                    print("q or c pressed, will quit", flush=True)
                    pygame.quit()
                    sys.exit()

            obs = unity_comms.getObservation()
            with open("obs.txt", "w") as file:
                file.write(obs)

            print(f'Obs {obs}')
            print(f'type of obs {type(obs)}')

            print(f'len of obs {len(obs)}')

            import PIL.Image as Image

            import io

            #bytes = bytearray(obs)
            b = bytes(obs, 'utf-8')

            #print(f'b {b}')
            print(f'type b {type(b)}')
            print(f'length of bytes {len(b)}')

            ba = bytearray(obs, encoding="ascii")
            print(f'type ba {type(ba)}')
            print(f'length of ba {len(ba)}')

            with open("imagepython.png", "wb") as file:
                file.write(b)

            sys.exit()

            #picture = Image.open(b)

            # display image
            # picture.show()

            #image = Image.frombytes('RGB', (240, 240), b, 'raw')
            # image.save("savepath")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
