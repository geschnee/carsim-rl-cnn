using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.IO;
using System.Linq;

// this script counts the obtained rewards and ends the episode if the time is up or another end event is triggered
// this responsibility was previously a part of the CheckpointManager and other scripts
public class EpisodeManager : MonoBehaviour
{

    // for debugging its public, then you can lift the TimeLimit in unity
    public float allowedTime = 20f; // TODO was 10f
    public float duration = 0f;

    public int reward_bootstrap_n = 4;

    private AIEngine aIEngine;
    private GameObject finishLine;

    //for data frame
    private List<double> velocities = new List<double>();
    private DataFrameManager df;

    public int passedGoals;

    private bool episodeRunning = false;

    private Vector3 lastPosition;



    private int step = -1;
    private float cumReward = 0f;
    private float distanceReward = 0f;
    private float velocityReward = 0f;
    public float rewardSinceLastGetReward = 0f;

    private string endEvent = "notEnded";
    private float lastDistance;

    public List<GameObject> centerIndicators = new List<GameObject>();

    private GameManager gameManager;

    // multiply by some constant, the reward is very small
    float distanceCoefficient = 10f;

    List<float> step_rewards = new List<float>();
    // the index indicates the step in which the reward was found

    public void PrepareAgent()
    {
        this.aIEngine = this.GetComponent<AIEngine>();

        gameManager = this.transform.parent.Find("GameManager").GetComponent<GameManager>();



        //PrintIndicators();
    }

    public void IncreaseSteps()
    {
        this.step++;
    }


    public void setCenterIndicators(List<GameObject> indicators)
    {
        this.centerIndicators = indicators;
    }

    public void StartEpisode()
    {
        Debug.LogWarning($"StartEpisode");
        this.duration = 0f;
        this.passedGoals = 0;
        this.cumReward = 0f;
        this.distanceReward = 0f;
        this.velocityReward = 0f;
        this.rewardSinceLastGetReward = 0f;
        this.lastPosition = this.transform.position;
        this.step = -1;

        this.PrepareAgent();

        //PrintIndicators();

        this.lastDistance = GetDistanceToNextGoal();

        //Debug.Log($"last distance in startepisode {this.lastDistance}");

        this.episodeRunning = true;
    }

    private void PrintIndicators()
    {

        Debug.Log($"centerIndicators found: {this.centerIndicators.Count}");

        return;
        /*for (int i = 0; i < this.centerIndicators.Count; i++)
        {
            Debug.Log($"centerIndicators[{i}] {this.centerIndicators[i]}");
            Debug.Log($"{this.centerIndicators[i].transform.parent.gameObject.name}");
        }*/
    }

    public void EndEpisode(string endEvent)
    {
        if (this.episodeRunning == false)
        {
            Debug.LogWarning($"EndEpisode called again {endEvent} before {this.endEvent}");
        }
        else
        {
            Debug.Log($"EndEpisode called {endEvent}");
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

    public string GetEndEvent()
    {
        if (this.episodeRunning == true)
        {
            Debug.LogWarning("GetEndEvent called but episode is still running");
            Debug.Log("GetEndEvent called but episode is still running");
        }


        return this.endEvent;
    }

    public Dictionary<string, string> GetInfo()
    {

        Dictionary<string, string> info = new Dictionary<string, string>();
        info.Add("endEvent", this.endEvent);
        info.Add("duration", this.duration.ToString());
        info.Add("cumreward", this.cumReward.ToString());
        info.Add("passedGoals", this.passedGoals.ToString());
        info.Add("distanceReward", this.distanceReward.ToString());
        info.Add("velocityReward", this.velocityReward.ToString());
        info.Add("step", this.step.ToString());
        info.Add("amount_of_steps", this.step_rewards.Count.ToString());

        info.Add("bootstrapped_rewards", this.GetBootstrappedRewards().ToString());


        return info;
    }

    private List<float> GetBootstrappedRewards()
    {
        List<float> bootstrapped_rewards = new List<float>();

        for (int i = 0; i < this.step_rewards.Count; i++)
        {
            float reward = 0f;
            for (int j = 0; j < this.reward_bootstrap_n; j++)
            {
                int index = i + j;
                if (index < this.step_rewards.Count)
                {
                    reward += this.step_rewards[index];
                }
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

        // add new entry for the step if it does not exist yet
        if (index >= this.step_rewards.Count)
        {
            this.step_rewards.Add(reward);
        }
        else
        {
            this.step_rewards[index] += reward;
        }
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
        //Debug.Log($"passedGoals {this.passedGoals} centerIndicators.Count {this.centerIndicators.Count}");

        //PrintIndicators();

        Vector3 nextGoal = this.centerIndicators[this.passedGoals].transform.position;
        Vector3 nextGoalDirection = nextGoal - this.transform.position;
        nextGoalDirection.y = 0; // set y difference to zero (we only care about the distance in the xz plane)
        // y is the horizontal difference

        return nextGoalDirection.magnitude;

    }

    public void FixedUpdate()
    {

        if (this.episodeRunning == false)
        {
            //Debug.Log("episode not running");
            return;
        }

        //small reward for speed
        // Debug.Log(this.drivingEngine.getCarVelocity() / 100f);

        this.duration += Time.deltaTime;


        //float distance = Vector3.Distance(this.gameObject.transform.localPosition, this.finishLine.transform.localPosition);
        //float distanceReward = 1 / distance;

        float velo = this.aIEngine.getCarVelocity();
        this.velocities.Add(velo);
        //Debug.Log(velo);
        // this.AddReward(distanceReward * Time.deltaTime);
        if (velo > 0)
        {
            AddVelocityReward((velo / 10f) * Time.deltaTime);

        }

        // reward for driving towards the next goal middleIndicator
        float distanceReward = this.lastDistance - GetDistanceToNextGoal();
        distanceReward *= this.distanceCoefficient;

        AddDistanceReward(distanceReward);
        // Debug.Log($"Distance reward: {distanceReward}");

        this.lastDistance = GetDistanceToNextGoal();


        if (this.duration >= this.allowedTime)
        {
            AddReward(-1f);

            string endEvent = "outOfTime";
            EndEpisode(endEvent);

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
        AddReward(100f);
        IncreasePassedGoals();

        EndEpisode("completeMap");
    }

    public void hitWall()
    {
        AddReward(-1f);


        string endEvent = "wall";
        EndEpisode(endEvent);
    }

    public void goalMissed()
    {
        AddReward(-1f);

        // TODO also save the color of the missed goal?
        string endEvent = "goalMissed";
        EndEpisode(endEvent);
    }

    public void goalPassed(GameObject goal)
    {

        AddTime(10f);

        AddReward(1f);

        IncreasePassedGoals();
        centerIndicators.RemoveAt(0); // remove an indicator

        Debug.Log($"will destroy goal {goal.name} {goal.transform.parent.gameObject.name}");
        Destroy(goal);

        // TODO also save the color of the completed goal?

        // update the distance to the next goal
        this.lastDistance = GetDistanceToNextGoal();

        Debug.Log($"completed a goal");
        //PrintIndicators();
    }

    public void obstacleHit()
    {
        AddReward(-1f);
    }

    public void redObstacleHit()
    {
        obstacleHit();
        string endEvent = "redObstacle";
        EndEpisode(endEvent);
    }

    public void blueObstacleHit()
    {
        obstacleHit();
        string endEvent = "blueObstacle";
        EndEpisode(endEvent);
    }


    // automatically detects when transform to which this is assigned hit another object with a tag
    private void OnTriggerEnter(Collider other)
    {
        // This is attached to the JetBot

        if (!this.episodeRunning)
        {
            Debug.LogWarning("Episode not running, ignore collision");
            return;
        }

        if (other.tag == "BlueObstacleTag")
        {
            blueObstacleHit();
        }
        if (other.tag == "RedObstacleTag")
        {
            redObstacleHit();
        }
        if (other.tag == "GoalPassed")
        {
            goalPassed(other.gameObject);
        }
        if (other.tag == "GoalMissed")
        {
            goalMissed();
        }
        if (other.tag == "Wall")
        {
            hitWall();
        }
        if (other.tag == "FinishCheckpoint")
        {
            finishCheckpoint();
        }
    }
}
