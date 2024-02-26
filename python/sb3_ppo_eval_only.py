

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv

import gymEnv.carsimGymEnv as carsimGymEnv

from myPPO.myPPO import myPPO

from gymEnv.carsimGymEnv import MapType
import torch
from torch.utils.tensorboard import SummaryWriter


import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


import logging
import os

def run_ppo_eval(cfg):
    logger = SummaryWriter(log_dir="./tmp")
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    logger.add_text("Configs", repr(cfg))
    logger.add_text("Torch Seed", f"{torch.get_rng_state()}")

    # we use own replay buffer that saves the observation space as uint8 instead of float32
    # int8 is 8bit, float32 is 32bit

    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

   

    assert cfg.copy_model_from != False, "copy_model_from must be set to a model path"
    

    n_envs = cfg.n_envs

    env_kwargs = cfg.env_kwargs
    env_kwargs["trainingMapType"] = MapType[cfg.env_kwargs.trainingMapType]
    # get proper enum type from string



    # Parallel environments
    vec_env = make_vec_env(carsimGymEnv.BaseCarsimEnv, n_envs=n_envs, env_kwargs=env_kwargs)
    # the n_envs can quickly be too much since the replay buffer will grow
    # the observations are quite big (float32)

    # set one log to true
    # ---> some logging of the stacked frames to see what the memory is like
    # there is some image printing to the imagelog folder
    vec_env.env_method(
        method_name="setLog",
        indices=0,
        log=True,
    )

    n_epochs, batch_size = cfg.n_epochs, cfg.batch_size
    n_steps = cfg.n_steps # amount of steps to collect per collect_rollouts per environment

    algo = myPPO # or myPPO (handles the ansynchronicity of the envs)


    print(f'using {algo} with {n_epochs} epochs, {batch_size} batch size and {n_steps} steps per epoch')
    policy_kwargs = {"normalize_images": cfg.env_kwargs.image_preprocessing.normalize_images}

    # normalize_imagess=True scales the images to 0-1 range
    # requires dtype float32
    # kwarg to both the env (ObsSpace) and the policy


    model = algo("CnnPolicy", vec_env, verbose=1,
                tensorboard_log="./tmp", n_epochs=n_epochs, batch_size=batch_size, n_steps=n_steps, policy_kwargs=policy_kwargs)
    # CnnPolicy network architecture can be seen in sb3.common.torch_layers.py


    
    if cfg.copy_model_from:
        string = f"{HydraConfig.get().runtime.cwd}/{cfg.copy_model_from}"
        print(f'loading model from {string} before learning')
        model = algo.load(string, env=vec_env, tensorboard_log="./tmp",
                        n_epochs=n_epochs, batch_size=batch_size)

    model.eval_only(total_eval_runs=cfg.eval_settings.number_eval_runs, num_evals_per_difficulty = cfg.eval_settings.num_evals_per_difficulty, eval_light_settings=cfg.eval_settings.eval_light_settings)


@hydra.main(config_path=".", config_name="cfg/ppo_eval.yaml")
def main(cfg):

    logging.info(f"cfg {cfg}")

    with open('config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    import cProfile
    cProfile.runctx('run_ppo_eval(cfg.cfg)', globals(), locals(), sort='cumtime')
    #run_ppo(cfg.cfg)

if __name__ == "__main__":
    main()