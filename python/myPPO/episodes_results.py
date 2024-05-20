
class EpisodesResults:

    completed_episodes, successfully_completed_episodes, number_of_goals, successfully_passed_goals, total_reward, total_timesteps, distance_reward, velocity_reward, event_reward, orientation_reward = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    timesteps_of_completed_episodes, collision_episodes = 0, 0
    obstacle_collision_episodes, wall_collision_episodes = 0, 0
    unity_duration = 0

    
    timeouts = 0
    successfully_passed_first_goals, successfully_passed_second_goals, successfully_passed_third_goals = 0, 0, 0
    waitTime = 0

    prescale_distance_reward, prescale_velocity_reward, prescale_event_reward, prescale_orientation_reward = 0, 0, 0, 0

    successful_episodes_with_collisions = 0

    num_easy_episodes, num_medium_episodes, num_hard_episodes = 0, 0, 0
    successful_easy_episodes, successful_medium_episodes, successful_hard_episodes = 0, 0, 0

    successful_easy_goals, successful_medium_goals, successful_hard_goals = 0, 0, 0
    easy_goals, medium_goals, hard_goals = 0, 0, 0


    def __init__(self):
        pass

    def processInfoDictEpisodeFinished(self, infos):

        self.completed_episodes += 1

        if infos["endEvent"] == "Success":
            self.successfully_completed_episodes += 1
        elif infos["endEvent"] == "OutOfTime":
            self.timeouts += 1
        else:
            #print(f'end event is {infos["endEvent"]}')
            # FinishMissed
            pass


        self.successfully_passed_goals += int(infos["passedGoals"])
        self.number_of_goals += int(infos["numberOfGoals"])
        self.total_reward += float(infos["cumreward"].replace(",","."))
        self.timesteps_of_completed_episodes += int(infos["amount_of_steps"])
        self.collision_episodes += int(infos["collision"])
        self.obstacle_collision_episodes += int(infos["obstacleCollision"])
        self.wall_collision_episodes += int(infos["wallCollision"])

        self.distance_reward += float(infos["distanceReward"].replace(",","."))
        self.velocity_reward += float(infos["velocityReward"].replace(",","."))
        self.event_reward += float(infos["eventReward"].replace(",","."))
        self.orientation_reward += float(infos["orientationReward"].replace(",","."))

        self.prescale_distance_reward += float(infos["prescaleDistanceReward"].replace(",","."))
        self.prescale_velocity_reward += float(infos["prescaleVelocityReward"].replace(",","."))
        self.prescale_event_reward += float(infos["prescaleEventReward"].replace(",","."))
        self.prescale_orientation_reward += float(infos["prescaleOrientationReward"].replace(",","."))

        self.successfully_passed_first_goals += int(infos["passedFirstGoal"])

        self.successfully_passed_second_goals += int(infos["passedSecondGoal"])
        self.successfully_passed_third_goals += int(infos["passedThirdGoal"])

        self.waitTime += float(infos["episodeWaitTime"])

        self.unity_duration += float(infos["duration"].replace(",","."))

        if int(infos["collision"]) == 1 and infos["endEvent"] == "Success":
            self.successful_episodes_with_collisions += 1

        if infos["mapDifficulty"] == "easy":
            self.num_easy_episodes += 1
            self.easy_goals += int(infos["numberOfGoals"])
            self.successful_easy_goals += int(infos["passedGoals"])
            if infos["endEvent"] == "Success":
                self.successful_easy_episodes += 1

        if infos["mapDifficulty"] == "medium":
            self.num_medium_episodes += 1
            self.medium_goals += int(infos["numberOfGoals"])
            self.successful_medium_goals += int(infos["passedGoals"])
            if infos["endEvent"] == "Success":
                self.successful_medium_episodes += 1
        if infos["mapDifficulty"] == "hard":
            self.num_hard_episodes += 1
            self.hard_goals += int(infos["numberOfGoals"])
            self.successful_hard_goals += int(infos["passedGoals"])
            if infos["endEvent"] == "Success":
                self.successful_hard_episodes += 1

    def computeRates(self):

        if self.number_of_goals != 0:
            self.goal_completion_rate = self.successfully_passed_goals / self.number_of_goals
        else:
            self.goal_completion_rate = 0
        
        if self.completed_episodes != 0:
            self.success_rate = self.successfully_completed_episodes / self.completed_episodes
            self.timeout_rate = self.timeouts / self.completed_episodes
            self.mean_reward = self.total_reward / self.completed_episodes
            self.mean_episode_length = self.timesteps_of_completed_episodes / self.completed_episodes
            self.mean_distance_reward = self.distance_reward / self.completed_episodes
            self.mean_velocity_reward = self.velocity_reward / self.completed_episodes
            self.mean_event_reward = self.event_reward / self.completed_episodes
            self.mean_orientation_reward = self.orientation_reward / self.completed_episodes
            self.first_goal_completion_rate = self.successfully_passed_first_goals / self.completed_episodes
            self.second_goal_completion_rate = self.successfully_passed_second_goals / self.completed_episodes
            self.third_goal_completion_rate = self.successfully_passed_third_goals / self.completed_episodes
            self.rate_episodes_with_collisions = self.collision_episodes / self.completed_episodes
            self.avg_step_duration_unity_env = self.unity_duration / self.timesteps_of_completed_episodes

            self.mean_prescale_distance_reward = self.prescale_distance_reward / self.completed_episodes
            self.mean_prescale_velocity_reward = self.prescale_velocity_reward / self.completed_episodes
            self.mean_prescale_event_reward = self.prescale_event_reward / self.completed_episodes
            self.mean_prescale_orientation_reward = self.prescale_orientation_reward / self.completed_episodes

            self.collision_rate = self.collision_episodes / self.completed_episodes
            self.obstacle_collision_rate = self.obstacle_collision_episodes / self.completed_episodes
            self.wall_collision_rate = self.wall_collision_episodes / self.completed_episodes

            if self.successfully_completed_episodes != 0:
                self.collision_rate_succesful_episodes = self.successful_episodes_with_collisions / self.successfully_completed_episodes
            else:
                self.collision_rate_succesful_episodes = 0

            self.rate_easy_episodes = self.num_easy_episodes / self.completed_episodes
            self.rate_medium_episodes = self.num_medium_episodes / self.completed_episodes
            self.rate_hard_episodes = self.num_hard_episodes / self.completed_episodes
            
        else:
            self.success_rate, self.mean_reward, self.mean_episode_length, self.mean_distance_reward, self.mean_velocity_reward, self.mean_event_reward, self.mean_orientation_reward = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.timeout_rate = 0
            self.rate_episodes_with_collisions = 0
            self.avg_step_duration_unity_env = 0
            self.mean_prescale_distance_reward, self.mean_prescale_velocity_reward, self.mean_prescale_event_reward, self.mean_prescale_orientation_reward = 0, 0, 0, 0

            self.collision_rate, self.obstacle_collision_rate, self.wall_collision_rate = 0, 0, 0

            self.collision_rate_succesful_episodes = 0

            self.first_goal_completion_rate, self.second_goal_completion_rate, self.third_goal_completion_rate = 0.0, 0.0, 0.0

            self.rate_easy_episodes, self.rate_medium_episodes, self.rate_hard_episodes = 0, 0, 0


        if self.num_easy_episodes != 0:
            self.easy_success_rate = self.successful_easy_episodes / self.num_easy_episodes
            self.easy_goal_completion_rate = self.successful_easy_goals / self.easy_goals
        else:
            self.easy_success_rate = 0
            self.easy_goal_completion_rate = 0
        if self.num_medium_episodes != 0:
            self.medium_success_rate = self.successful_medium_episodes / self.num_medium_episodes
            self.medium_goal_completion_rate = self.successful_medium_goals / self.medium_goals
        else:
            self.medium_success_rate = 0
            self.medium_goal_completion_rate = 0
        if self.num_hard_episodes != 0:
            self.hard_success_rate = self.successful_hard_episodes / self.num_hard_episodes
            self.hard_goal_completion_rate = self.successful_hard_goals / self.hard_goals
        else:
            self.hard_success_rate = 0
            self.hard_goal_completion_rate = 0
        