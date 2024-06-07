

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import set_random_seed


from stable_baselines3.common.vec_env import DummyVecEnv

import gymEnv.carsimGymEnv as carsimGymEnv

from myPPO.myPPO import myPPO

from gymEnv.myEnums import MapType
from gymEnv.myEnums import LightSetting
from gymEnv.myEnums import SpawnOrientation

import torch
from torch.utils.tensorboard import SummaryWriter


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


import logging
import os
import random

def run_replay(cfg):
    logger = SummaryWriter(log_dir="./tmp")
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    logger.add_text("Configs", repr(cfg))

    seed = cfg.seed
    
    print(f"Torch Seed before {torch.get_rng_state()}")

    set_random_seed(seed)
    # https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility
    logger.add_text("Torch Seed", f"{torch.get_rng_state()}")

    print(f'a random torch int {torch.randint(0, 100, (1,))}')
    print(f'a random torch int2 {torch.randint(0, 100, (1,))}')

    print(f'a random int {random.randint(0, 100)}')
    print(f'a random int2 {random.randint(0, 100)}')
    
    print(f"Torch Seed after {torch.get_rng_state()}")

    # we use own replay buffer that saves the observation space as uint8 instead of float32
    # int8 is 8bit, float32 is 32bit

    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")


    n_envs = cfg.n_envs

    env_kwargs = OmegaConf.to_container(cfg.env_kwargs)
    env_kwargs["trainingMapType"] = MapType[cfg.env_kwargs.trainingMapType]
    env_kwargs["trainingLightSetting"] = LightSetting[cfg.env_kwargs.trainingLightSetting]
    # get proper enum type from string
    env_kwargs["spawnOrientation"] = SpawnOrientation[cfg.env_kwargs.spawnOrientation]



    # Parallel environments
    vec_env = make_vec_env(carsimGymEnv.BaseCarsimEnv, n_envs=n_envs, env_kwargs=env_kwargs)
    # the n_envs can quickly be too much since the replay buffer will grow
    # the observations are quite big (float32)


    algo = myPPO

    policy_kwargs = {"net_arch": OmegaConf.to_container(cfg.algo_settings.net_arch)}

    model = algo(cfg.algo_settings.policy, vec_env, verbose=1,
                tensorboard_log="./tmp", n_epochs=cfg.algo_settings.n_epochs, batch_size=cfg.algo_settings.batch_size, n_steps=cfg.algo_settings.n_steps, policy_kwargs=policy_kwargs, seed = seed, use_bundled_calls=cfg.algo_settings.use_bundled_calls, use_fresh_obs=cfg.algo_settings.use_fresh_obs, print_network_and_loss_structure=cfg.algo_settings.print_network_and_loss_structure)
    # CnnPolicy network architecture can be seen in sb3.common.torch_layers.py


    assert cfg.episode_record_replay_settings.replay_folder, "replay_folder must be set in the config file"

    # I am not sure why but without this model load it does not produce the right outputs
    string = f'{HydraConfig.get().runtime.cwd}/{cfg.episode_record_replay_settings.replay_folder}/model'
    print(f'loading model from {string} before replay', flush=True)
    model = algo.load(string, env=vec_env, tensorboard_log="./tmp", n_epochs=cfg.algo_settings.n_epochs, batch_size=cfg.algo_settings.batch_size)
 
    model.replay_episodes(cfg.episode_record_replay_settings, seed, cfg.env_kwargs.fixedTimestepsLength)


@hydra.main(config_path=".", config_name="cfg/ppo_replay_only.yaml")
def main(cfg):

    # run specific config files with:
    # python sb3_ppo.py --config-name cfg/ppo_isolated_medium_standard.yaml

    logging.info(f"cfg {cfg}")

    with open('config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    #import cProfile
    #cProfile.runctx('run_replay(cfg.cfg)', globals(), locals(), sort='cumtime')
    run_replay(cfg.cfg)

if __name__ == "__main__":
    main()