# from https://github.com/hughperkins/peaceful-pie/blob/main/examples/SimpleNetworking/python/get_height.py

import argparse

from peaceful_pie.unity_comms import UnityComms

import time


def run(args: argparse.Namespace) -> None:
    print(f'will try port {args.port}', flush=True)
    unity_comms = UnityComms(port=args.port)
    print(f'Unity comms created', flush=True)
    t = time.time()
    res = unity_comms.getHeight()
    print(f'Request took {time.time() - t} seconds')
    print("res", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    run(args)
