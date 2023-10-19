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
    public float allowedTime = 10f;
    public float duration = 0f;

    private AIEngine aIEngine;
    private GameObject finishLine;

    //for data frame
    private List<double> velocities = new List<double>();
    private DataFrameManager df;
    private int steps;

    private int passedGoals;

    private bool terminated = false;
    private bool episodeRunning = false;

    private Vector3 lastPosition;

    private float cumReward = 0f;
    private float rewardSinceLastGetReward = 0f;

    private string endEvent = "notEnded";
    private float lastDistance;

    private List<GameObject> centerIndicators = new List<GameObject>();



    public void Awake()
    {
        this.aIEngine = this.GetComponent<AIEngine>();

        GameObject allGoals = this.transform.parent.Find("AllGoals").gameObject;
        Debug.Log($"allGoals {allGoals}");
        Debug.Log($"allGoals child amount {allGoals.transform.childCount}");

        this.centerIndicators = new List<GameObject>();

        for (int i = 0; i < allGoals.transform.childCount; i++)
        {
            GameObject goal = allGoals.transform.GetChild(i).gameObject;
            Debug.Log($"goal {goal}");

            GameObject middle = FindChildWithTag(goal, "GoalPassed");
            GameObject middleFinished = FindChildWithTag(goal, "FinishCheckpoint");
            if (middle != null)
            {
                this.centerIndicators.Add(middle);
            }
            if (middleFinished != null)
            {
                this.centerIndicators.Add(middleFinished);
            }
        }

        Debug.Log($"centerIndicators found: {this.centerIndicators.Count}");
    }

    private GameObject FindChildWithTag(GameObject parent, string tag)
    {
        Transform t = parent.transform;
        for (int i = 0; i < t.childCount; i++)
        {
            if (t.GetChild(i).gameObject.tag == tag)
            {
                return t.GetChild(i).gameObject;
            }
        }

        return null;
    }

    public void StartEpisode()
    {

        this.duration = 0f;
        this.passedGoals = 0;
        this.terminated = false;
        this.cumReward = 0f;
        this.rewardSinceLastGetReward = 0f;
        this.lastPosition = this.transform.position;

        this.episodeRunning = true;

        this.lastDistance = GetDistanceToNextGoal();
    }

    public void EndEpisode(string endEvent)
    {
        if (this.terminated)
        {
            Debug.LogWarning($"something went wrong, episode already terminated but EpisodeManager.EndEpisode called again {endEvent}");
        }

        this.terminated = true;
        this.endEvent = endEvent;
        this.episodeRunning = false;
    }

    public float GetReward()
    {
        float r = this.rewardSinceLastGetReward;

        this.rewardSinceLastGetReward = 0f;

        return r;
    }

    public bool IsTerminated()
    {
        return this.terminated;
    }

    public string GetEndEvent()
    {
        if (this.terminated == false)
            Debug.LogWarning("GetEndEvent called but episode not terminated yet");

        return this.endEvent;
    }


    public void AddReward(float reward)
    {
        this.cumReward += reward;
        this.rewardSinceLastGetReward += reward;
    }

    public float GetDistanceToNextGoal()
    {
        Vector3 nextGoal = this.centerIndicators[this.passedGoals].transform.position;
        Vector3 nextGoalDirection = nextGoal - this.transform.position;
        nextGoalDirection.y = 0; // set y difference to zero (we only care about the distance in the xz plane)
        return nextGoalDirection.magnitude;

    }

    public void FixedUpdate()
    {

        if (this.episodeRunning == false)
        {
            Debug.Log("episode not running");
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
            AddReward((velo / 10f) * Time.deltaTime);

        }

        // reward for driving towards the next goal middleIndicator
        float distanceReward = this.lastDistance - GetDistanceToNextGoal();
        AddReward(distanceReward);
        Debug.Log($"Distance reward: {distanceReward}");

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
        Destroy(goal);

        // TODO also save the color of the completed goal?

        // update the distance to the next goal
        this.lastDistance = GetDistanceToNextGoal();



        IncreasePassedGoals();
    }

    public void obstacleHit()
    {
        AddReward(-1f);
    }

    public void redObstacleHit()
    {
        string endEvent = "redObstacle";
        EndEpisode(endEvent);
        obstacleHit();
    }

    public void blueObstacleHit()
    {
        string endEvent = "blueObstacle";
        EndEpisode(endEvent);
        obstacleHit();
    }


    // automatically detects when transform to which this is assigned hit another object with a tag
    private void OnTriggerEnter(Collider other)
    {
        //        Debug.Log("Triggered by: " + other.tag);
        //      Debug.Log("Chackpoint manager attached to " + this.name);

        // This is attached to the JetBot

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
