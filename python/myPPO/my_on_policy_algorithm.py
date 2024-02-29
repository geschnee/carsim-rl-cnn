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

from gymEnv.myEnums import LightSetting

import os

SelfOnPolicyAlgorithm = TypeVar(
    "SelfOnPolicyAlgorithm", bound="MyOnPolicyAlgorithm")

# what this code is based on:
# https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/on_policy_algorithm.html
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

        print(f'collect rollouts started', flush=True)
        cr_time = time.time() 

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        completed_episodes, successfully_completed_episodes, number_of_goals, successfully_passed_goals, total_reward, total_timesteps, distance_reward, velocity_reward, event_reward, orientation_reward, successfully_passed_first_goals = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        timesteps_of_completed_episodes, collision_episodes = 0, 0

        # event_reward is the reward that is not distance or velocity reward
        # such as collisions, passed goals and episode terminated
        timeouts = 0
        second_goals_given_first, third_goals_given_second, successfully_passed_second_goals = 0, 0, 0
        waitTime = 0

        reward_correction_dict = {}
        for i in range(env.num_envs):
            reward_correction_dict[i] = {}
        # outer dictionary maps env index to inner dictionary
        # inner dictionary maps step number to the corresponding position in rollout_buffer

        
        env.reset()
        # we need to reset the env to get the correct rewards

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                fresh_obs = get_obs(env)
                obs_tensor = obs_as_tensor(fresh_obs, self.device)

                
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
        

            self.num_timesteps += env.num_envs
            total_timesteps += env.num_envs

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
                    completed_episodes += 1
                
                    if infos[idx]["endEvent"] == "Success":
                        successfully_completed_episodes += 1
                    if infos[i]["endEvent"] == "OutOfTime":
                        timeouts += 1

                    successfully_passed_goals += int(infos[idx]["passedGoals"])
                    number_of_goals += int(infos[idx]["numberOfGoals"])
                    total_reward += float(infos[idx]["cumreward"].replace(",","."))
                    timesteps_of_completed_episodes += int(infos[idx]["amount_of_steps"])
                    collision_episodes += int(infos[idx]["collision"])
                    

                    distance_reward += float(infos[idx]["distanceReward"].replace(",","."))
                    velocity_reward += float(infos[idx]["velocityReward"].replace(",","."))
                    event_reward += float(infos[idx]["eventReward"].replace(",","."))
                    orientation_reward += float(infos[idx]["orientationReward"].replace(",","."))
                    successfully_passed_first_goals += int(infos[idx]["passedFirstGoal"])

                    successfully_passed_second_goals += int(infos[idx]["passedSecondGoal"])

                    waitTime += float(infos[idx]["episodeWaitTime"])
                    
                    if int(infos[idx]["passedFirstGoal"]) == 1 and int(infos[idx]["passedSecondGoal"]) == 1:
                        second_goals_given_first +=1
                    if int(infos[idx]["passedSecondGoal"]) == 1 and int(infos[idx]["passedThirdGoal"]) == 1:
                        third_goals_given_second +=1

            insertpos = rollout_buffer.add(
                fresh_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            #print(f'insertpos: {insertpos}')

            assert len(infos) == env.num_envs, f"infos has wrong length {len(infos)} != {env.num_envs}"

            for idx, info in enumerate(infos):

                assert int(info['step']) not in reward_correction_dict[idx].keys(), f"step {info['step']} already in reward correction dict for env {idx}"
                if int(info['step']) != 0:
                    assert int(info["step"]) == max(reward_correction_dict[idx].keys()) + 1, f"step {info['step']} is not the next step in reward correction dict for env {idx}: {reward_correction_dict[idx]}"
                
                reward_correction_dict[idx][int(info['step'])] = insertpos

            for idx, done in enumerate(dones):
                # obtain the real rewards from the env
                if done:
                    self.collected_episodes += 1

                if done or n_steps >= n_rollout_steps:
                    # playout finished or cancelling due to enough collected datapoints
                    # TODO it would be better to remove the ones prematurely terminated from the replay buffer

                    env_id = idx
                    rewards = infos[env_id]['rewards']
                    assert len(rewards) == len(reward_correction_dict[env_id]), f"rewards {len(rewards)} and reward correction dict {len(reward_correction_dict[env_id])} do not match in length"
                    for step, bufferpos in reward_correction_dict[env_id].items():
                        rollout_buffer.rewards[bufferpos][env_id] = rewards[step]


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

        
        if number_of_goals != 0:
            goal_completion_rate = successfully_passed_goals / number_of_goals
        else:
            goal_completion_rate = 0
        
        if completed_episodes != 0:
            success_rate = successfully_completed_episodes / completed_episodes
            timeout_rate = timeouts / completed_episodes
            mean_reward = total_reward / completed_episodes
            mean_episode_length = timesteps_of_completed_episodes / completed_episodes
            mean_distance_reward = distance_reward / completed_episodes
            mean_velocity_reward = velocity_reward / completed_episodes
            mean_event_reward = event_reward / completed_episodes
            mean_orientation_reward = orientation_reward / completed_episodes
            first_goal_completion_rate = successfully_passed_first_goals / completed_episodes
            rate_episodes_with_collisions = collision_episodes / completed_episodes
        else:
            success_rate, mean_reward, mean_episode_length, mean_distance_reward, mean_velocity_reward, mean_event_reward, mean_orientation_reward, first_goal_completion_rate = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            timeout_rate = 0
            rate_episodes_with_collisions = 0
        
        
        step_average_wait_time = waitTime / total_timesteps


        if successfully_passed_goals > 0:
            print(f'passed a goal succesfully, rate is {goal_completion_rate}')
            assert goal_completion_rate > 0.0, "goal completion rate is 0 although a goal was passed"
        
        if successfully_passed_first_goals > 0:
            rate_second_goal_given_first = second_goals_given_first / successfully_passed_first_goals
        else:
            rate_second_goal_given_first = 0
        if successfully_passed_second_goals > 0:
            rate_third_goal_given_second = third_goals_given_second / successfully_passed_second_goals
        else:
            rate_third_goal_given_second = 0
        self.logger.record("rollout/rate_second_given_first", rate_second_goal_given_first)
        self.logger.record("rollout/rate_third_given_second", rate_third_goal_given_second)
        
        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/timeout_rate", timeout_rate)
        self.logger.record("rollout/goal_completion_rate", goal_completion_rate)
        self.logger.record("rollout/mean_reward", mean_reward)
        self.logger.record("rollout/mean_episode_length", mean_episode_length)
        self.logger.record("rollout/mean_distance_reward", mean_distance_reward)
        self.logger.record("rollout/mean_velocity_reward", mean_velocity_reward)
        self.logger.record("rollout/mean_orientation_reward", mean_orientation_reward)
        self.logger.record("rollout/mean_event_reward", mean_event_reward)
        self.logger.record("rollout/first_goal_completion_rate", first_goal_completion_rate)
        self.logger.record("rollout/completed_episodes", completed_episodes)
        self.logger.record("rollout/step_average_wait_time", step_average_wait_time)
        self.logger.record("rollout/rate_episodes_with_collisions", rate_episodes_with_collisions)
        

        cr_time = time.time() - cr_time
        
        print(f'collect rollouts finished with {completed_episodes} episodes in {cr_time} seconds', flush=True)

        return True, cr_time
    

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
        num_evals_per_difficulty: int = 10,
        eval_light_settings: bool = False,
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

        total_cr_time, total_train_time, total_eval_time = 0, 0, 0
        self.collected_episodes = 0
        self.max_total_success_rate = 0

        total_collection_time = 0

        while self.num_timesteps < total_timesteps:
            # collect_rollouts
            
            should_log = log_interval is not None and iteration % log_interval == 0
            continue_training, cr_time = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            
            total_collection_time += cr_time

            total_cr_time += cr_time

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(
                self.num_timesteps, total_timesteps)


            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                # log interval is 0, thus after every ollect rollouts the tb logging is done
                # the log x axis is self.num_timesteps, which is modified in collect_rollouts


                assert self.ep_info_buffer is not None
                time_elapsed = max(
                    (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                # what is fps?
                # num_timestep is increased in collect_rollouts for n_env after each step
                # The time/fps are thus distributed for the n_envs
                # computed in real-time not simulation time
                fps_per_env = float(fps / self.n_envs)

                # fps takes the time for the whole training, collect_rollout_fps_per_env only counts collection time
                collect_rollout_fps_per_env = float((self.num_timesteps - self._num_timesteps_at_start) / total_collection_time)

                self.logger.record("time/iterations",
                                   iteration, exclude="tensorboard")
                '''if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(
                        "rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))'''
                self.logger.record("time/fps", fps)
                self.logger.record("time/fps_per_env", fps_per_env)
                self.logger.record("time/collect_rollout_fps_per_env", collect_rollout_fps_per_env)

                self.logger.record("time/time_elapsed",
                                   int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps",
                                   self.num_timesteps, exclude="tensorboard")
                self.logger.record("rollout/collected_episodes", self.collected_episodes)

                self.logger.record("time/collection_time_seconds", cr_time)
                self.logger.record("time/iteration", iteration)
                

                self.logger.dump(step=self.num_timesteps)

            train_time = time.time()
            self.train()
            train_time = time.time() - train_time
            self.logger.record("time/train_time_seconds", train_time)
            
            self.logger.dump(step=self.num_timesteps)
            total_train_time += train_time

            # model eval 
            if log_interval is not None and iteration % log_interval == 0:
                print(f'Will eval now as after every {log_interval} collect and trains')
                eval_time = time.time()
                self.eval(iteration=iteration, num_evals_per_difficulty=num_evals_per_difficulty, eval_light_settings=eval_light_settings)

                
                eval_time = time.time() - eval_time
                self.logger.record("time/eval_time_seconds", eval_time)

                print(f'eval finished minutes: {eval_time / 60}')
                total_eval_time += eval_time

                self.logger.dump(step=self.num_timesteps)

                print(f'total_cr_time: {total_cr_time}')
                print(f'total_train_time: {total_train_time}')
                print(f'total_eval_time: {total_eval_time}', flush=True)
            

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    
    def eval(self: SelfOnPolicyAlgorithm, iteration: int = 0, num_evals_per_difficulty: int = 20, eval_light_settings: bool = False) -> float:
        print(f'eval started', flush=True)



        if eval_light_settings:
            light_settings = [LightSetting.bright, LightSetting.standard, LightSetting.dark]
        else: 
            light_settings = [LightSetting.standard]

        os.mkdir(f'{os.getcwd()}\\videos_iter_{iteration}')

        total_success_rate = 0

        avg_easy_success_rate, avg_medium_success_rate, avg_hard_success_rate = 0, 0, 0

        for light_setting in light_settings:
            
            easy_success_rate = self.eval_model_track(num_evals_per_difficulty, "easy", iteration, light_setting)
            medium_success_rate = self.eval_model_track(num_evals_per_difficulty, "medium", iteration, light_setting)
            hard_success_rate = self.eval_model_track(num_evals_per_difficulty, "hard", iteration, light_setting)
            total_success_rate += easy_success_rate + medium_success_rate + hard_success_rate
            light_success_rate = (easy_success_rate + medium_success_rate + hard_success_rate) / 3
            
            self.logger.record(f"eval/success_easy_light_{light_setting.name}", easy_success_rate)
            self.logger.record(f"eval/success_medium_light_{light_setting.name}", medium_success_rate)
            self.logger.record(f"eval/success_hard_light_{light_setting.name}", hard_success_rate)
            self.logger.record(f"eval/success_light_{light_setting.name}", light_success_rate)
            avg_easy_success_rate += easy_success_rate
            avg_medium_success_rate += medium_success_rate
            avg_hard_success_rate += hard_success_rate

        if eval_light_settings:
            self.logger.record(f"eval/success_easy_across_light_settings", avg_easy_success_rate / len(light_settings))
            self.logger.record(f"eval/success_medium_across_light_settings", avg_medium_success_rate / len(light_settings))
            self.logger.record(f"eval/success_hard_across_light_settings", avg_hard_success_rate / len(light_settings))

        total_success_rate = total_success_rate / (3 * len(light_settings))
        if total_success_rate > self.max_total_success_rate:
            self.max_total_success_rate = total_success_rate
            self.save(f"best_model_episode_{iteration}")


        self.logger.record("eval/success_rate", total_success_rate)

        return total_success_rate

    def eval_model_track(
        self: SelfOnPolicyAlgorithm,
        n_eval_episodes: int = 10,
        difficulty: str = "easy",
        iteration: int = 0,
        light_setting: LightSetting = LightSetting.standard,
    ):
        # spawn position and orientation is defined by the env via cfg.env_kwargs.spawn_point

        env = self.env
        n_envs = env.num_envs
        episode_rewards = []
        episode_lengths = []
        success_count, finished_episodes, passed_goals, number_of_goals = 0, 0, 0, 0
        first_goals, second_goals, third_goals = 0, 0, 0
        second_goals_given_first, third_goals_given_second = 0, 0
        collision_episodes = 0

        timeouts = 0
        wait_time = 0


        episode_counts = np.zeros(n_envs, dtype="int")
        # Divides episodes among different sub environments in the vector as evenly as possible
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
        # episode_count_targets represents the amount of episodes that have to be played in the corresponding env
        # the sum of these values is equal to n_eval_episodes

        current_rewards = np.zeros(n_envs)
        current_lengths = np.zeros(n_envs, dtype="int")


        # reset environment 0 to record the videos
        log_indices = [0]
        for i in log_indices:
            env.env_method(
                method_name="setVideoFilename",
                indices=[i],
                video_filename = f'{os.getcwd()}\\videos_iter_{iteration}\\{difficulty}_{light_setting.name}_env_{i}_video_'
            )
        
        env.env_method(
            method_name="reset_with_difficulty",
            indices=range(n_envs),
            difficulty=difficulty,
            lightSetting=light_setting,
        )
        
        # switch to eval mode
        self.policy.set_training_mode(False)

        while (episode_counts < episode_count_targets).any():
    
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                fresh_obs = get_obs(env) 
                obs_tensor = obs_as_tensor(fresh_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high)
            
            observations, rewards, dones, infos = env.step(clipped_actions)
            
            current_lengths += 1
            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:

                    if dones[i]:
                        
                        episode_rewards.append(float(infos[i]["cumreward"].replace(",",".")))
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                        current_rewards[i] = 0
                        current_lengths[i] = 0
                        finished_episodes += 1
                        passed_goals += int(infos[i]["passedGoals"])
                        number_of_goals += int(infos[i]["numberOfGoals"])


                        #print(f'collision_episodes {collision_episodes} + {int(infos[i]["collision"])}')
                        collision_episodes += int(infos[i]["collision"])

                        first_goals += int(infos[i]["passedFirstGoal"])
                        #if int(infos[i]["passedGoals"]) >= 2:
                        second_goals += int(infos[i]["passedSecondGoal"])
                        #if int(infos[i]["passedGoals"]) >= 3:
                        third_goals += int(infos[i]["passedThirdGoal"])

                        wait_time += float(infos[i]["episodeWaitTime"])

                        if int(infos[i]["passedFirstGoal"]) == 1 and int(infos[i]["passedSecondGoal"]) == 1:
                            second_goals_given_first +=1
                        if int(infos[i]["passedSecondGoal"]) == 1 and int(infos[i]["passedThirdGoal"]) == 1:
                            third_goals_given_second +=1

                        

                        if infos[i]["endEvent"] == "Success":
                            success_count += 1
                        if infos[i]["endEvent"] == "OutOfTime":
                            timeouts += 1

                        if i in log_indices:
                            if episode_counts[i] == episode_count_targets[i]-1:
                                # no more logging needed for this env
                                env.env_method(
                                    method_name="setVideoFilename",
                                    indices=[i],
                                    video_filename = ""
                                )


                        # due to auto reset we have to reset the env again with the right parameters:
                        env.env_method(
                            method_name="reset_with_difficulty",
                            indices=[i],
                            difficulty=difficulty,
                            lightSetting=light_setting
                        )

        assert np.sum(episode_counts) == n_eval_episodes, f"not all episodes were finished, {np.sum(episode_counts)} != {n_eval_episodes}"
        assert finished_episodes == n_eval_episodes, f"not all episodes were finished, {finished_episodes} != {n_eval_episodes}"
        

        #print(f'episode_rewards: {episode_rewards}')
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        success_rate = success_count / n_eval_episodes
        rate_of_passed_goals = passed_goals / number_of_goals
        rate_of_passed_first_goals = first_goals / n_eval_episodes
        if first_goals > 0:
            rate_of_second_goal_given_first = second_goals_given_first / first_goals
        else:
            rate_of_second_goal_given_first = 0
        if second_goals > 0:
            rate_of_third_goal_given_second = third_goals_given_second / second_goals
        else:
            rate_of_third_goal_given_second = 0

        timeout_rate = timeouts/n_eval_episodes



        self.logger.record(f'eval_{difficulty}_{light_setting.name}/mean_reward', mean_reward)
        
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/std_reward', std_reward)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/success_rate', success_rate)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/rate_passed_goals', rate_of_passed_goals)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/rate_first_goal', rate_of_passed_first_goals)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/rate_second_goal_given_first', rate_of_second_goal_given_first)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/rate_third_goal_given_second', rate_of_third_goal_given_second)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/timeout_rate', timeout_rate)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/rate_episode_with_collision', collision_episodes / n_eval_episodes)

        step_average_wait_time = wait_time / np.sum(episode_lengths)
        self.logger.record(f"eval_{difficulty}_{light_setting.name}/step_average_wait_time", step_average_wait_time)
        self.logger.record(f'eval_{difficulty}_{light_setting.name}/average_episode_length', np.average(episode_lengths))

        # set to no video afterwards
        env.env_method(
            method_name="setVideoFilename",
            indices=[0],
            video_filename = ""
        )

        return success_rate
    

    

    def eval_only(
        self: SelfOnPolicyAlgorithm,
        total_eval_runs: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        num_evals_per_difficulty: int = 10,
        eval_light_settings: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        
        from stable_baselines3.common import utils
        self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        assert self.env is not None

        total_eval_time = 0

        for i in range(total_eval_runs):
            eval_time = time.time()
            self.eval(iteration=iteration, num_evals_per_difficulty=num_evals_per_difficulty, eval_light_settings=eval_light_settings)

            
            eval_time = time.time() - eval_time
            self.logger.record("time/eval_time_seconds", eval_time)

            print(f'eval finished minutes: {eval_time / 60}')
            total_eval_time += eval_time

            self.logger.dump(step=i)

            print(f'total_eval_time: {total_eval_time}', flush=True)

        return self

def get_obs(env):
    # env is a vectorized BaseCarsimEnv
    # it is wrapped in a vec_transpose env for the CNN

    
    # TODO this does a memory rollover, as well as the step function
    # should one of them not do it? or does it not matter?
    # the frequent rollovers might result in the history not being deep enough

    # I think the cleanest way to solve this is by simply increasing frame stacking n

    for idx in range(env.num_envs):
        obs = env.envs[idx].get_observation_including_memory()
        # get_obseration_including memory does a memory rolloer as well
        env._save_obs(idx, obs)

    obs = env._obs_from_buf()
    return env.transpose_observations(obs)