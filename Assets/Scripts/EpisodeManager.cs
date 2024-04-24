using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;

public class EpisodeManager : MonoBehaviour
{
    // needed for fixed timesteps
    private float timeOfLastStepBegin = 0;
    public bool fixedTimesteps;
    public float fixedTimestepsLength;
    public bool stepFinished;

    // for debugging its public, then you can lift the TimeLimit in unity
    private float allowedTimeDefault = 20f; // was 10f
    private float allowedTimePerGoal = 40f; // was 10f

    // multiply by some constant, the reward is very small
    public float distanceCoefficient;// = 10f;
    public float velocityCoefficient;// = 0.1f;
    public float orientationCoefficient;// = 0.1f;
    public float eventCoefficient;// = 1f;

    private float finishCheckpointReward = 1f; // = 100f;
    private float wallHitReward = -1f;
    private float obstacleHitReward = -1f;
    private float timeoutReward = -1f;
    private float goalMissedReward = -1f;
    private float goalPassedReward = 1f;
    public float duration;
    public float allowedTime;
    public bool obstacleOrWallHit;

    private AIEngineBase aIEngine;
    private GameObject finishLine;

    public List<int> passedGoals;
    public int numberOfGoals;

    public EpisodeStatus episodeStatus;

    private Vector3 lastPosition;

    private GameObject carBody;
    public GameObject distanceReference;

    public int step;
    private float cumReward;
    private float distanceReward;
    private float velocityReward;
    private float orientationReward;
    private float eventReward;

    private float prescaleDistanceReward;
    private float prescaleVelocityReward;
    private float prescaleOrientationReward;
    private float prescaleEventReward;

    private float lastDistance;

    public List<GameObject> centerIndicators = new List<GameObject>();

    private GameManager gameManager;
    public VideoRecorder arenaRecorder;
    public VideoRecorder jetBotRecorder;

    List<float> step_rewards;
    // the index indicates the step in which the reward was found

    public void PrepareAgent()
    {
        this.aIEngine = this.GetComponent<AIEngineBase>();

        gameManager = this.transform.parent.Find("GameManager").GetComponent<GameManager>();
        carBody = this.transform.Find("carBody").gameObject;

    }

    public void IncreaseSteps(int step)
    {


        // int step from python is not yet incremented
        if (this.step != step)
        {
            // Debug.LogWarning($"unity step {this.step} != python step {step} for {this.transform.parent.name}");
            // this.step != step can happen when there is a timeout in python-unity communication (the message is sent again)
            // it also happens when the bundledSteps are used and some steps were not finished --> leeds to a resend of the step instructions

            if (this.step < step)
            {
                Debug.LogError($"unity step {this.step} > python step {step} for {this.transform.parent.name}");
                // this should not happen, as python controlls the steps and unity follows behind
            }

            return;
        }

        this.timeOfLastStepBegin = Time.time;

        this.step++;
        // add new entry in the rewards counting list
        // with this added 0 reward there is an entry for every step even if there was no reward signal encountered


        this.step_rewards.Add(0);
        if (this.step_rewards.Count != (this.step + 1))
        {
            Debug.LogError($"count should be one higher than steps: step_rewards.Count {this.step_rewards.Count} step {this.step}");

        }

        if (this.episodeStatus == EpisodeStatus.WaitingForStep)
        {
            if (!this.fixedTimesteps && this.step != 0)
            {
                Debug.LogWarning($"IncreaseSteps called while waiting for step {this.step} {this.episodeStatus} but we do not use fixed timesteps, there must be some implementation error");
            }
            this.episodeStatus = EpisodeStatus.Running;
            this.stepFinished = false;
        }

    }


    public void setCenterIndicators(List<GameObject> indicators)
    {
        this.centerIndicators = indicators;
        this.numberOfGoals = indicators.Count;
    }

    public void StartEpisode()
    {
        this.duration = 0f;
        this.passedGoals = new List<int>();
        this.cumReward = 0f;
        this.distanceReward = 0f;
        this.velocityReward = 0f;
        this.eventReward = 0f;
        this.orientationReward = 0f;
        this.prescaleDistanceReward = 0f;
        this.prescaleVelocityReward = 0f;
        this.prescaleEventReward = 0f;
        this.prescaleOrientationReward = 0f;
        this.lastPosition = this.transform.position;
        this.step = -1;
        this.allowedTime = this.allowedTimeDefault;
        this.obstacleOrWallHit = false;

        initializeStepRewards();

        this.PrepareAgent();

        this.lastDistance = GetDistanceToNextGoal();

        aIEngine.ResetMotor();


        this.stepFinished = false;
        this.episodeStatus = EpisodeStatus.WaitingForStep; // has to wait for the first step command
    }

    private void initializeStepRewards()
    {
        this.step_rewards = new List<float>();
    }

    public void EndEpisode(EpisodeStatus episodeStatus)
    {
        if (IsTerminated())
        {
            Debug.LogWarning($"EndEpisode called again {episodeStatus} before {this.episodeStatus}");
        }
        this.episodeStatus = episodeStatus;

        arenaRecorder.StopVideo(this.duration);
        jetBotRecorder.StopVideo(this.duration);

        this.stepFinished = true;

    }

    public float GetReward()
    {
        float r = this.step_rewards[this.step];

        return r;
    }

    public bool IsTerminated()
    {
        return (this.episodeStatus != EpisodeStatus.Running) && (this.episodeStatus != EpisodeStatus.WaitingForStep);
    }

    private string GetEndEvent()
    {
        if (IsTerminated() == false)
        {
            Debug.LogWarning($"GetEndEvent called but episode is still running {this.episodeStatus}");
            Debug.Log("GetEndEvent called but episode is still running");
        }

        return this.episodeStatus.ToString();
    }

    public Dictionary<string, string> GetInfo()
    {
        Dictionary<string, string> info = new Dictionary<string, string>();
        info.Add("endEvent", this.episodeStatus.ToString());
        info.Add("duration", this.duration.ToString());
        info.Add("cumreward", this.cumReward.ToString());
        info.Add("passedGoals", this.passedGoals.Count.ToString());
        info.Add("passedFirstGoal", this.passedGoals.Contains(0) ? "1" : "0");
        info.Add("passedSecondGoal", this.passedGoals.Contains(1) ? "1" : "0");
        info.Add("passedThirdGoal", this.passedGoals.Contains(2) ? "1" : "0");

        info.Add("numberOfGoals", this.numberOfGoals.ToString());
        info.Add("distanceReward", this.distanceReward.ToString());
        info.Add("orientationReward", this.orientationReward.ToString());
        info.Add("eventReward", this.eventReward.ToString());
        info.Add("velocityReward", this.velocityReward.ToString());
        info.Add("prescaleDistanceReward", this.prescaleDistanceReward.ToString());
        info.Add("prescaleOrientationReward", this.prescaleOrientationReward.ToString());
        info.Add("prescaleEventReward", this.prescaleEventReward.ToString());
        info.Add("prescaleVelocityReward", this.prescaleVelocityReward.ToString());
        info.Add("step", this.step.ToString());
        info.Add("amount_of_steps", (this.step + 1).ToString());
        info.Add("amount_of_steps_based_on_rewardlist", this.step_rewards.Count.ToString());

        info.Add("collision", this.obstacleOrWallHit ? "1" : "0");

        return info;
    }

    public List<float> GetRewards()
    {
        /*List<float> rewards = new List<float>();

        int amount_of_steps = this.step + 1;
        for (int i = 0; i < amount_of_steps; i++)
        {
            float reward = this.step_rewards[i];
            rewards.Add(reward);
        }
        return rewards;*/

        return this.step_rewards;
    }

    public bool isEpisodeRunning()
    {
        return this.episodeStatus == EpisodeStatus.Running;
    }


    public void AddReward(float reward)
    {
        if (!isEpisodeRunning()) // reward that is obtained before a step is performed is given to the first step
        {
            Debug.LogError($"Adding reward while episode is not running {this.step}, {this.episodeStatus}");
            return;
        }

        this.cumReward += reward;

        int index;

        index = this.step;

        this.step_rewards[index] += reward;

    }

    public void AddEventReward(float reward)
    {

        if (reward<0)
        {
            Debug.LogError($"negative reward {reward}, will return now");
            // this is just a test
            // it could be that the agent never tries to go through goals as the collision risk is too high
            // a collision might punish too much, as there might be multiple collision triggers in one single timestep/collision
            return;
        }

        this.prescaleEventReward += reward;
        float weightedEventReward = reward * eventCoefficient;
        this.eventReward += weightedEventReward;
        AddReward(weightedEventReward);
    }

    public void AddDistanceReward(float reward)
    {
        this.prescaleDistanceReward += reward;
        float weightedDistanceReward = reward * distanceCoefficient;
        this.distanceReward += weightedDistanceReward;
        // Debug.Log($"AddDistanceReward {weightedDistanceReward} {this.distanceReward}");
        AddReward(weightedDistanceReward);
    }

    public void AddOrientationReward(float reward)
    {
        this.prescaleOrientationReward += reward;
        float weightedOrientationReward = reward * orientationCoefficient;
        this.orientationReward += weightedOrientationReward;
        AddReward(weightedOrientationReward);
    }

    public void AddVelocityReward(float reward)
    {
        this.prescaleVelocityReward += reward;
        float weightedVelocityReward = reward * velocityCoefficient;
        this.velocityReward += weightedVelocityReward;
        AddReward(weightedVelocityReward);
    }

    public float GetDistanceToNextGoal()
    {

        if (this.centerIndicators.Count < 1)
        {
            Debug.LogWarning($"{this.gameObject.name} there are no more centerIndicators {this.centerIndicators.Count} episodeStatus {this.episodeStatus} passedGoals {this.passedGoals}");

            return 0f;
        }


        Vector3 nextGoal = this.centerIndicators[0].transform.position;
        Vector3 nextGoalDirection = nextGoal - this.distanceReference.transform.position;
        nextGoalDirection.y = 0;
        // set y difference to zero (we only care about the distance in the xz plane)
        // y is the horizontal difference


        return nextGoalDirection.magnitude;
    }

    public float GetCosineSimilarityToNextGoal()
    {



        if (this.centerIndicators.Count < 1)
        {
            Debug.LogError($"there are no more centerIndicators {this.gameObject.name}");
            return 0f;
        }

        Vector3 nextGoal = this.centerIndicators[0].transform.position;
        Vector3 nextGoalDirection = nextGoal - carBody.transform.position;
        nextGoalDirection.y = 0;
        // set y difference to zero (we only care about the distance in the xz plane)
        // y is the horizontal difference

        Vector3 agentOrientation = this.transform.forward;




        float angleBetween = Vector3.Angle(agentOrientation, nextGoalDirection);
        //Debug.Log($"agentOrientation {agentOrientation} nextGoalDirection {nextGoalDirection}");
        //Debug.Log($"angle between {angleBetween}");

        float cosine_similarity = GetCosineSimilarityXZPlane(nextGoalDirection, agentOrientation);

        //Debug.Log($"cosine_similarity XZ plane {cosine_similarity}");

        return cosine_similarity;
    }

    public static float GetCosineSimilarityXZPlane(Vector3 V1, Vector3 V2)
    {
        float result = (float)((V1.x * V2.x + V1.z * V2.z)
                / (Math.Sqrt(Math.Pow(V1.x, 2) + Math.Pow(V1.z, 2)) *
                    Math.Sqrt(Math.Pow(V2.x, 2) + Math.Pow(V2.z, 2))
                ));

        if (result > 1)
        {
            Debug.LogError("cosine sim too big");
        }
        if (result < -1)
        {
            Debug.LogError("cosine sim too small");
        }
        return result;
    }

    public void FixedUpdate()
    {
        if (this.duration >= this.allowedTime && this.episodeStatus == EpisodeStatus.Running)
        {
            AddEventReward(timeoutReward);
            EndEpisode(EpisodeStatus.OutOfTime);
            return;
        }

        if (this.fixedTimesteps && isEpisodeRunning())
        {
            if (Time.time - this.timeOfLastStepBegin > this.fixedTimestepsLength)
            {
                this.episodeStatus = EpisodeStatus.WaitingForStep;
                // fixed timesteps and the time of the current step is up

                this.stepFinished = true;
            }
        }

        if (!isEpisodeRunning())
        {
            return;
        }



        // count time only when it is running
        this.duration += Time.deltaTime;



        float velo = this.aIEngine.getCarVelocity();

        AddVelocityReward((velo) * Time.deltaTime);
        

        // reward for driving towards the next goal middleIndicator
        float distanceReward = this.lastDistance - GetDistanceToNextGoal();

        AddDistanceReward(distanceReward);

        // reward for orientation in direction of next goal
        float orientationReward = GetCosineSimilarityToNextGoal() * Time.deltaTime;
        AddOrientationReward(orientationReward);

        this.lastDistance = GetDistanceToNextGoal();




    }

    public void AddTime(float time)
    {
        this.allowedTime += time;
    }

    public float getTimeSinceEpisodeStart()
    {
        return this.duration;
    }

    public void IncreasePassedGoals(GameObject goalMiddle)
    {
        GameObject goal = goalMiddle.transform.parent.gameObject;
        int goalnumber = goal.transform.name[goal.transform.name.Length - 1] - '0';
        this.passedGoals.Add(goalnumber);
    }


    public void finishCheckpoint(GameObject goal)
    {
        AddEventReward(finishCheckpointReward);
        IncreasePassedGoals(goal);

        GameObject goal2 = goal.transform.parent.gameObject;
        colorGreen(goal2);

        EndEpisode(EpisodeStatus.Success);
    }

    public void hitWall()
    {
        AddEventReward(wallHitReward);
        collision();
    }

    public void destroyCheckpoint(GameObject goal)
    {
        // Destroy all child objects of the goal that were used for checking the goal pass
        // this is needed since we do not want to end the collision on missed goals anymore

        Transform t = goal.transform;
        for (int i = 0; i < t.childCount; i++)
        {
            string tag = t.GetChild(i).gameObject.tag;
            if (tag == "GoalPassed" | tag == "GoalMissed" | tag == "Destroyed")
            {
                t.GetChild(i).gameObject.tag = "Destroyed";
                Destroy(t.GetChild(i).gameObject);
            }

        }
    }



    public void colorGreen(GameObject goal)
    {

        Transform t = goal.transform;
        for (int i = 0; i < t.childCount; i++)
        {
            string tag = t.GetChild(i).gameObject.tag;
            if (tag == "GoalBall")
            {
                t.GetChild(i).GetComponent<BallColor>().SetGreen();
            }
        }
    }

    public void colorRed(GameObject goal)
    {

        Transform t = goal.transform;
        for (int i = 0; i < t.childCount; i++)
        {
            string tag = t.GetChild(i).gameObject.tag;
            if (tag == "GoalBall")
            {
                t.GetChild(i).GetComponent<BallColor>().SetRed();
            }
        }
    }

    public void finishMissed(GameObject border)
    {
        GameObject goal = border.transform.parent.gameObject;
        destroyCheckpoint(goal);
        AddEventReward(goalMissedReward);

        centerIndicators.RemoveAt(0); // remove an indicator

        colorRed(goal);

        EndEpisode(EpisodeStatus.FinishMissed);
    }

    public void goalPassed(GameObject goalMiddle)
    {

        IncreasePassedGoals(goalMiddle);

        GameObject goal = goalMiddle.transform.parent.gameObject;
        destroyCheckpoint(goal);
        AddEventReward(goalPassedReward);

        AddTime(allowedTimePerGoal);

        centerIndicators.RemoveAt(0); // remove an indicator

        // update the distance to the next goal
        this.lastDistance = GetDistanceToNextGoal();

        colorGreen(goal);
    }

    public void goalMissed(GameObject redBorder)
    {
        GameObject goal = redBorder.transform.parent.gameObject;
        destroyCheckpoint(goal);
        AddEventReward(goalMissedReward);

        centerIndicators.RemoveAt(0); // remove an indicator


        this.lastDistance = GetDistanceToNextGoal();


        colorRed(goal);
    }

    public void obstacleHit()
    {
        AddEventReward(obstacleHitReward);
        collision();
    }

    public void collision()
    {
        this.obstacleOrWallHit = true;
    }

    public void redObstacleHit()
    {
        obstacleHit();
    }

    public void blueObstacleHit()
    {
        obstacleHit();
    }

    private void OnTriggerStay(Collider coll)
    {
        Collision(coll);
    }

    private void OnTriggerEnter(Collider coll)
    {
        Collision(coll);
    }

    private void Collision(Collider coll)
    {
        // This is attached to the JetBot

        if (!isEpisodeRunning())
        {
            //Debug.LogWarning("Episode not running, ignore collision with " + coll.tag);
            return;
        }
        if (coll.tag == "BlueObstacleTag")
        {
            blueObstacleHit();
            return;
        }
        if (coll.tag == "RedObstacleTag")
        {
            redObstacleHit();
            return;
        }
        if (coll.tag == "GoalPassed")
        {
            coll.tag = "Destroyed";
            goalPassed(coll.gameObject);
            return;
        }
        if (coll.tag == "GoalMissed")
        {
            goalMissed(coll.gameObject);
            return;
        }
        if (coll.tag == "Wall")
        {
            hitWall();

            return;
        }
        if (coll.tag == "FinishMissed")
        {
            finishMissed(coll.gameObject);
            return;
        }
        if (coll.tag == "FinishCheckpoint")
        {
            finishCheckpoint(coll.gameObject);
            return;
        }
        if (coll.tag == "Destroyed")
        {
            // duplicate detection, ignore
            return;
        }
        Debug.LogError($"unknown tag {coll.tag}");
    }
}
