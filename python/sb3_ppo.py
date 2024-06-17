

# this RLLIB could also be interesting
# https://docs.ray.io/en/latest/rllib/index.html
# this requires python 3.9, currently installed 3.11 on laptop

import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.utils import set_random_seed


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

def run_ppo(cfg):
    logger = SummaryWriter(log_dir="./tmp")
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    logger.add_text("Configs", repr(cfg))

    seed = cfg.seed
    
    print(f"Torch Seed before {torch.get_rng_state()}")

    set_random_seed(seed)
    # https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility
    logger.add_text("Torch Seed", f"{torch.get_rng_state()}")

    # we use own replay buffer that saves the observation space as uint8 instead of float32
    # int8 is 8bit, float32 is 32bit --> this saves a lot of space

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

    
    vec_env.env_method(
        method_name="setSeedUnity",
        indices=0,
        seed = seed,
    )


    # set one log to true
    # ---> some logging of the stacked frames to see what the memory is like
    # there is some image printing to the imagelog folder
    vec_env.env_method(
        method_name="setLog",
        indices=0,
        log=False,
    )

    algo = myPPO

    policy_kwargs = {"net_arch": OmegaConf.to_container(cfg.algo_settings.net_arch)}

    model = algo(cfg.algo_settings.policy, vec_env, verbose=1,
                tensorboard_log="./tmp", n_epochs=cfg.algo_settings.n_epochs, batch_size=cfg.algo_settings.batch_size, n_steps=cfg.algo_settings.n_steps, policy_kwargs=policy_kwargs, seed = seed, use_bundled_calls=cfg.algo_settings.use_bundled_calls, use_fresh_obs=cfg.algo_settings.use_fresh_obs, print_network_and_loss_structure=cfg.algo_settings.print_network_and_loss_structure)
    # CnnPolicy network architecture can be seen in sb3.common.torch_layers.py

    print(f"model weights for seed verification: {model.policy.value_net.weight[0][0:5]}")
    # [-0.0591, -0.0703,  0.0513, -0.1466,  0.0055] for seed 2048

    
    if cfg.copy_model_from:
        string = f"{HydraConfig.get().runtime.cwd}/{cfg.copy_model_from}"
        print(f'loading model from {string} before learning', flush=True)
        model = algo.load(string, env=vec_env, tensorboard_log="./tmp",
                        n_epochs=cfg.algo_settings.n_epochs, batch_size=cfg.algo_settings.batch_size)

    if not cfg.eval_settings.eval_only:
        model.learn(total_timesteps=cfg.total_timesteps, n_eval_episodes = cfg.eval_settings.n_eval_episodes, eval_light_settings=cfg.eval_settings.eval_light_settings)
        model.save("finished_ppo")
        print("finished learning without issues")

    
    # load best model and eval it again
    if not cfg.eval_settings.eval_only:
        best_model_name = model.rollout_best_model_name
        print(f'loading best model {best_model_name} after learning')
        model.load(best_model_name)
        model.use_bundled_calls = cfg.algo_settings.use_bundled_calls
        model.use_fresh_obs=cfg.algo_settings.use_fresh_obs

    
    # run more evals here after training completed or when eval only
    model.eval_only(total_eval_runs=cfg.eval_settings.number_eval_runs, n_eval_episodes = cfg.eval_settings.n_eval_episodes, eval_light_settings=cfg.eval_settings.eval_light_settings, offset=model.num_timesteps)

    if not cfg.episode_record_replay_settings.replay_folder:
        model.record_episodes(cfg.episode_record_replay_settings, seed, cfg)
    else:
        print(f"replaying episodes from {cfg.episode_record_replay_settings.replay_folder}")
    

    if cfg.episode_record_replay_settings.replay_folder:
        model_path = os.path.join(HydraConfig.get().runtime.cwd, cfg.episode_record_replay_settings.replay_folder, "model")
    else:
        # replay the ones that were previously recorded in this same run
        model_path = os.path.join(os.getcwd(), "episode_recordings", "model")

    print(f'loading model from {model_path} before replay', flush=True)
    model = algo.load(model_path, env=vec_env, tensorboard_log="./tmp", n_epochs=cfg.algo_settings.n_epochs, batch_size=cfg.algo_settings.batch_size)

    model.replay_episodes(cfg.episode_record_replay_settings, seed, cfg.env_kwargs.fixedTimestepsLength)
    


@hydra.main(config_path=".", config_name="cfg/ppo.yaml")
def main(cfg):
    # run specific config files with:
    # python sb3_ppo.py --config-name cfg/ppo_isolated_medium_standard.yaml

    logging.info(f"cfg {cfg}")

    with open('config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    import cProfile
    cProfile.runctx('run_ppo(cfg.cfg)', globals(), locals(), sort='cumtime')
    #run_ppo(cfg.cfg)

if __name__ == "__main__":
    main()