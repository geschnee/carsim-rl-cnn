using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.IO;
using System.Linq;


public enum EndEvent
{
    NotEnded = 0,
    Success = 1,
    OutOfTime = 2,
    WallHit = 3,
    GoalMissed = 4,
    RedObstacle = 5,
    BlueObstacle = 6
}

// this script counts the obtained rewards and ends the episode if the time is up or another end event is triggered
// this responsibility was previously a part of the CheckpointManager and other scripts
public class EpisodeManager : MonoBehaviour
{

    // for debugging its public, then you can lift the TimeLimit in unity
    public float allowedTimeDefault = 20f; // TODO was 10f
    public float allowedTimePerGoal = 10f; // TODO was 10f

    // multiply by some constant, the reward is very small
    float distanceCoefficient = 10f;

    public float finishCheckpointReward = 100f;
    public float wallHitReward = -1f;
    public float obstacleHitReward = -1f;
    public float timeoutReward = -1f;
    public float goalMissedReward = -1f;
    public float goalPassedReward = 1f;

    int reward_bootstrap_n; // this is set by python env
    float reward_bootstrap_discount = 0.9f;

    public float duration;
    public float allowedTime;

    private AIEngine aIEngine;
    private GameObject finishLine;


    private DataFrameManager df;

    public int passedGoals;

    private bool episodeRunning = false;
    private EndEvent endEvent;

    private Vector3 lastPosition;



    private int step;
    private float cumReward;
    private float distanceReward;
    private float velocityReward;
    private float otherReward;
    public float rewardSinceLastGetReward;


    private float lastDistance;

    public List<GameObject> centerIndicators = new List<GameObject>();

    private GameManager gameManager;

    List<float> step_rewards;
    // the index indicates the step in which the reward was found

    public void PrepareAgent()
    {
        this.aIEngine = this.GetComponent<AIEngine>();

        gameManager = this.transform.parent.Find("GameManager").GetComponent<GameManager>();
    }

    public void SetBootstrapN(int n)
    {
        this.reward_bootstrap_n = n;
    }

    public void IncreaseSteps()
    {
        this.step++;
        // add new entry in the rewards counting list
        // with this added 0 reward there is an entry for every step even if there was no reward signal encountered
        if (this.step != 0)
        {
            this.step_rewards.Add(0);

        }
        if (this.step_rewards.Count != (this.step + 1))
        {
            Debug.LogError($"count should be one higher than steps: step_rewards.Count {this.step_rewards.Count} step {this.step}");

        }

    }


    public void setCenterIndicators(List<GameObject> indicators)
    {
        this.centerIndicators = indicators;
    }

    public void StartEpisode()
    {
        this.duration = 0f;
        this.passedGoals = 0;
        this.cumReward = 0f;
        this.distanceReward = 0f;
        this.velocityReward = 0f;
        this.otherReward = 0f;
        this.rewardSinceLastGetReward = 0f;
        this.lastPosition = this.transform.position;
        this.step = -1;
        this.allowedTime = this.allowedTimeDefault;
        this.endEvent = EndEvent.NotEnded;
        initializeStepRewards();

        this.PrepareAgent();

        this.lastDistance = GetDistanceToNextGoal();

        this.episodeRunning = true;
    }

    private void initializeStepRewards()
    {
        this.step_rewards = new List<float>();
        this.step_rewards.Add(0); // for rewards for step 0
    }

    public void EndEpisode(EndEvent endEvent)
    {
        if (this.episodeRunning == false)
        {
            Debug.LogWarning($"EndEpisode called again {endEvent} before {this.endEvent}");
        }
        else
        {
            //Debug.Log($"EndEpisode called {endEvent}");
        }

        this.episodeRunning = false;
        this.endEvent = endEvent;

        aIEngine.ResetMotor();
        aIEngine.episodeRunning = false;
    }

    public float GetReward()
    {
        float r = this.rewardSinceLastGetReward;

        this.rewardSinceLastGetReward = 0f;

        return r;
    }

    public bool IsTerminated()
    {
        return !this.episodeRunning;
    }

    private string GetEndEvent()
    {
        if (this.episodeRunning == true)
        {
            Debug.LogWarning("GetEndEvent called but episode is still running");
            Debug.Log("GetEndEvent called but episode is still running");
        }

        return this.endEvent.ToString();
    }

    public Dictionary<string, string> GetInfo()
    {
        Dictionary<string, string> info = new Dictionary<string, string>();
        info.Add("endEvent", this.endEvent.ToString());
        info.Add("duration", this.duration.ToString());
        info.Add("cumreward", this.cumReward.ToString());
        info.Add("passedGoals", this.passedGoals.ToString());
        info.Add("numberOfGoals", this.centerIndicators.Count.ToString());
        info.Add("distanceReward", this.distanceReward.ToString());
        info.Add("otherReward", this.otherReward.ToString());
        info.Add("velocityReward", this.velocityReward.ToString());
        info.Add("step", this.step.ToString());
        info.Add("amount_of_steps", (this.step + 1).ToString());
        info.Add("amount_of_steps_based_on_rewardlist", this.step_rewards.Count.ToString());

        return info;
    }

    public List<float> GetBootstrappedRewards()
    {
        List<float> bootstrapped_rewards = new List<float>();

        int amount_of_steps = this.step + 1;
        for (int i = 0; i < amount_of_steps; i++)
        {
            float reward = 0f;
            for (int j = 0; j < this.reward_bootstrap_n; j++)
            {
                int index = i + j;
                if (index == amount_of_steps)
                {
                    break; // there are no more steps/rewards to bootstrap
                }

                // do we need some kind of discount for bootstrapped rewards?
                // yes i think so, see the RL book
                reward += (float)Math.Pow((double)this.reward_bootstrap_discount, (double)index) * this.step_rewards[index];

            }
            bootstrapped_rewards.Add(reward);
        }
        return bootstrapped_rewards;
    }


    public void AddReward(float reward)
    {
        this.cumReward += reward;
        this.rewardSinceLastGetReward += reward;

        int index;
        if (this.step == -1) // reward that is obtained before a step is performed is given to the first step
        {
            index = 0;
        }
        else
        {
            index = this.step;
        }
        this.step_rewards[index] += reward;

    }

    public void AddOtherReward(float reward)
    {
        this.otherReward += reward;
        AddReward(reward);
    }

    public void AddDistanceReward(float reward)
    {
        this.distanceReward += reward;
        AddReward(reward);
    }

    public void AddVelocityReward(float reward)
    {
        this.velocityReward += reward;
        AddReward(reward);
    }

    public float GetDistanceToNextGoal()
    {

        if (this.passedGoals >= this.centerIndicators.Count)
        {
            return 0f;
        }


        Vector3 nextGoal = this.centerIndicators[this.passedGoals].transform.position;
        Vector3 nextGoalDirection = nextGoal - this.transform.position;
        nextGoalDirection.y = 0;
        // set y difference to zero (we only care about the distance in the xz plane)
        // y is the horizontal difference


        return nextGoalDirection.magnitude;
    }

    public void FixedUpdate()
    {

        if (this.episodeRunning == false)
        {

            return;
        }

        this.duration += Time.deltaTime;


        float velo = this.aIEngine.getCarVelocity();

        if (velo > 0)
        {
            AddVelocityReward((velo / 10f) * Time.deltaTime);
        }

        // reward for driving towards the next goal middleIndicator
        float distanceReward = this.lastDistance - GetDistanceToNextGoal();
        distanceReward *= this.distanceCoefficient;
        AddDistanceReward(distanceReward);


        this.lastDistance = GetDistanceToNextGoal();

        if (this.duration >= this.allowedTime)
        {
            AddReward(timeoutReward);
            EndEpisode(EndEvent.OutOfTime);
        }
    }

    public void AddTime(float time)
    {
        this.allowedTime += time;
    }

    public float getTimeSinceEpisodeStart()
    {
        return this.duration;
    }

    public void IncreasePassedGoals()
    {
        this.passedGoals++;
    }


    public void finishCheckpoint()
    {
        AddOtherReward(finishCheckpointReward);
        IncreasePassedGoals();

        EndEpisode(EndEvent.Success);
    }

    public void hitWall()
    {
        AddOtherReward(wallHitReward);



        EndEpisode(EndEvent.WallHit);
    }

    public void goalMissed()
    {
        AddOtherReward(goalMissedReward);

        // TODO also save the color of the missed goal?
        EndEpisode(EndEvent.GoalMissed);
    }

    public void goalPassed(GameObject goal)
    {

        AddTime(allowedTimePerGoal);

        AddOtherReward(goalPassedReward);

        IncreasePassedGoals();
        centerIndicators.RemoveAt(0); // remove an indicator

        //Debug.Log($"will destroy goal {goal.name} {goal.transform.parent.gameObject.name}");
        Destroy(goal);

        // TODO also save the color of the completed goal?

        // update the distance to the next goal
        this.lastDistance = GetDistanceToNextGoal();
    }

    public void obstacleHit()
    {
        AddOtherReward(obstacleHitReward);
    }

    public void redObstacleHit()
    {
        obstacleHit();
        EndEpisode(EndEvent.RedObstacle);
    }

    public void blueObstacleHit()
    {
        obstacleHit();
        EndEpisode(EndEvent.BlueObstacle);
    }

    private void OnTriggerEnter(Collider other)
    {
        // This is attached to the JetBot

        if (!this.episodeRunning)
        {
            //Debug.LogWarning("Episode not running, ignore collision with " + other.tag);
            return;
        }
        if (other.tag == "BlueObstacleTag")
        {
            blueObstacleHit();
            return;
        }
        if (other.tag == "RedObstacleTag")
        {
            redObstacleHit();
            return;
        }
        if (other.tag == "GoalPassed")
        {
            goalPassed(other.gameObject);
            return;
        }
        if (other.tag == "GoalMissed")
        {
            goalMissed();
            return;
        }
        if (other.tag == "Wall")
        {
            hitWall();
            return;
        }
        if (other.tag == "FinishCheckpoint")
        {
            finishCheckpoint();
            return;
        }
        Debug.LogError($"unknown tag {other.tag}");
    }
}
