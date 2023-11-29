import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from myPPO.my_buffers import MyRolloutBuffer

SelfOnPolicyAlgorithm = TypeVar(
    "SelfOnPolicyAlgorithm", bound="MyOnPolicyAlgorithm")


class MyOnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: MyRolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(
            self.observation_space, spaces.Dict) else MyRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MyRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        print(f'collect rollouts started')


        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        completed_games, successfully_completed_games, number_of_goals, successfully_passed_goals, total_reward, total_timesteps, distance_reward, velocity_reward, other_reward = 0, 0, 0, 0, 0, 0, 0, 0, 0
        # other reward is the reward that is not distance or velocity reward
        # such as collisions, passed goals and episode terminated

        reward_correction_dict = {}
        for i in range(env.num_envs):
            reward_correction_dict[i] = {}
        # outer dictionary maps env index to inner dictionary
        # inner dictionary maps step number to the corresponding position in rollout_buffer

        env.reset()
        # we need to reset the env to get the correct bootstrapped rewards

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                fresh_obs = get_obs(env)
                #print(f'fresh obs shape: {fresh_obs.shape}')
                #print(f'last obs shape: {self._last_obs.shape}')
                obs_tensor = obs_as_tensor(fresh_obs, self.device)
                #print(f'obs tensor shape: {obs_tensor.shape}')

                
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # TODO use the correct step method
            
            
            #print(f'new obs shape: {new_obs.shape}')
            # is 10, 3, 84, 84

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

                if done:
                    completed_games += 1
                
                    if infos[idx]["endEvent"] == "success":
                        successfully_completed_games += 1
                    successfully_passed_goals += int(infos[idx]["passedGoals"])
                    number_of_goals += int(infos[idx]["numberOfGoals"])
                    total_reward += float(infos[idx]["cumreward"])
                    total_timesteps += int(infos[idx]["amount_of_steps"])

                    distance_reward += float(infos[idx]["distanceReward"])
                    velocity_reward += float(infos[idx]["velocityReward"])
                    other_reward += float(infos[idx]["otherReward"])
                    

            insertpos = rollout_buffer.add(
                fresh_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            print(f'insertpos: {insertpos}')

            assert len(infos) == env.num_envs, f"infos has wrong length {len(infos)} != {env.num_envs}"

            for idx, info in enumerate(infos):
                # TODO is the returned step sometimes not correct?
                # returns the same step multiple times
                

                assert int(info['step']) not in reward_correction_dict[idx].keys(), f"step {info['step']} already in reward correction dict for env {idx}"
                if int(info['step']) != 0:
                    assert int(info["step"]) == max(reward_correction_dict[idx].keys()) + 1, f"step {info['step']} is not the next step in reward correction dict for env {idx}: {reward_correction_dict[idx]}"
                
                reward_correction_dict[idx][int(info['step'])] = insertpos

                # TODO es passiert nicht, dass etwas mehrmals in reward_correction_dict eingetragen wird
                # (es wird nicht ueberschrieben), es muss ein anderes Problem sein

                # reward correction dict fuer index 7 war kaputt:
                # reward correction dict: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {0: 76, 2: 77, 3: 78, 4: 79}, 8: {0: 76, 1: 77, 2: 78, 3: 79}, 9: {0: 76, 1: 77, 2: 78, 3: 79}}



            for idx, done in enumerate(dones):
                # TODO obtain the real rewards from the env
                # these rewards should be available from the info dictionary since the env is automatically reset when done

                # TODO the env from unity does not yet return a proper list/dict of rewards

                if done or n_steps >= n_rollout_steps:
                    # playout finished or cancelling due to enough collected datapoints
                    # TODO it would be better to remove the ones prematurely terminated from the replay buffer

                    print(f'reward correction dict: {reward_correction_dict}')
                    print(f'reward correction dict entry {reward_correction_dict[idx]}')

                    print(f'info for index {idx}: {infos[idx]}')

                    

                    env_id = idx
                    bootstrapped_rewards = infos[env_id]['bootstrapped_rewards']
                    assert len(bootstrapped_rewards) == len(reward_correction_dict[env_id]), f"bootstrapped rewards {len(bootstrapped_rewards)} and reward correction dict {len(reward_correction_dict[env_id])} do not match in length"
                    for step, bufferpos in reward_correction_dict[env_id].items():
                        rollout_buffer.rewards[bufferpos][env_id] = bootstrapped_rewards[step]


                    assert len(reward_correction_dict[env_id]) == int(infos[idx]['amount_of_steps']), f"reward correction dict is not complete {len(reward_correction_dict[env_id])} != {infos[idx]['amount_of_steps']}"

                    # reset the reward correction dict
                    reward_correction_dict[env_id] = {}

            
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(
                new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones)

        callback.on_rollout_end()

        assert completed_games>0, "not a single playout was complete, finished too fast"

        success_rate = successfully_completed_games / completed_games
        goal_completion_rate = successfully_passed_goals / number_of_goals

        if successfully_passed_goals > 0:
            print(f'passed a goal succesfully, rate is {goal_completion_rate}')
            assert goal_completion_rate > 0.0, "goal completion rate is 0 although a goal was passed"

        mean_reward = total_reward / completed_games
        mean_episode_length = total_timesteps / completed_games
        mean_distance_reward = distance_reward / completed_games
        mean_velocity_reward = velocity_reward / completed_games
        mean_other_reward = other_reward / completed_games
        return True, success_rate, goal_completion_rate, mean_reward, mean_episode_length, mean_distance_reward, mean_velocity_reward, mean_other_reward
    

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training, success_rate, goal_completion_rate, mean_reward, mean_episode_length, mean_distance_reward, mean_velocity_reward, mean_other_reward = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:

                # TODO check how ep_info_buffer is filled
                # is it accurate although we do our special bootstrapping?
                # no it is not, it is filled from the directly returned rewards
                # not from the rewards in the info dict at final step


                assert self.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations",
                                   iteration, exclude="tensorboard")
                '''if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(
                        "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))'''
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed",
                                   int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps",
                                   self.num_timesteps, exclude="tensorboard")
                
                self.logger.record("rollout/success_rate", success_rate)
                self.logger.record("rollout/goal_completion_rate", goal_completion_rate)
                self.logger.record("rollout/mean_reward", mean_reward)
                self.logger.record("rollout/mean_episode_length", mean_episode_length)
                self.logger.record("rollout/mean_distance_reward", mean_distance_reward)
                self.logger.record("rollout/mean_velocity_reward", mean_velocity_reward)
                self.logger.record("rollout/mean_other_reward", mean_other_reward) # rewards such as collisions, goals passed and timeouts
                # this reward should not negatively dominate the other rewards
                # the penalties for expired time and collisions should not discourage the agent from moving

                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

def get_obs(env):
    # env is a vectorized BaseUnityCarEnv
    # it is wrapped in a vec_transpose env for the CNN

    # this also changes the observation stored in the dummy_vec_env

    for idx in range(env.num_envs):
        obs = env.envs[idx].get_observation_including_memory()
        #print(f'get_obs obs shape: {obs.shape}')
        env._save_obs(idx, obs)

    obs = env._obs_from_buf()
    return env.transpose_observations(obs)
