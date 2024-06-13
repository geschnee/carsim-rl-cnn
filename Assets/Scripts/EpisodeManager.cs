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

    private float allowedTimeDefault = 30f; // was 10f
    private float allowedTimePerGoal = 30f; // was 10f

    // multiply by some constant, the reward is very small
    public float distanceCoefficient;// = 10f;
    public float velocityCoefficient;// = 0.1f;
    public float orientationCoefficient;// = 0.1f;
    public float eventCoefficient;// = 1f;

    private float finishLineReward = 100f; // = 100f;
    private float wallHitReward = -1f;
    private float obstacleHitReward = -1f;
    private float timeoutReward = -1f;
    private float goalMissedReward = -1f;
    private float goalPassedReward = 100f;
    public float duration;
    public float allowedTime;
    public bool obstacleOrWallHit;
    public bool obstacleHit;
    public bool wallHit;

    private AIEngineBase aIEngine;
    private GameObject finishLine;

    public List<int> passedGoals;
    public int numberOfGoals;

    public EpisodeStatus episodeStatus;
    public CollisionMode collisionMode;
    public bool evalMode;
    public bool timestepObstacleHit;

    public List<GameObject> hitObstacles = new List<GameObject>();

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

    private MapManager gameManager;
    public VideoRecorder arenaRecorder;
    public VideoRecorder topViewRecorder;
    public VideoRecorder jetBotRecorder;

    List<float> step_rewards;
    // the index indicates the step in which the reward was found

    private MapType mapType;
    private string jetbot_name;
    private bool finishLineHit;

    public void PrepareAgent()
    {
        this.aIEngine = this.GetComponent<AIEngineBase>();

        gameManager = this.transform.parent.Find("MapManager").GetComponent<MapManager>();
        carBody = this.transform.Find("carBody").gameObject;
    }

    public void IncreaseSteps(int step)
    {
        // int step from python is not yet incremented
        if (this.step != step)
        {
            // this.step != step can happen when there is a timeout in python-unity communication (the message is sent again)
            // it also happens when the bundledSteps are used and some steps were not finished --> leeds to a resend of the step instructions

            if (this.step < step)
            {
                Debug.LogError($"unity step {this.step} > python step {step} for {this.transform.parent.name}");
                // this should not happen, as python controlls the steps and unity follows behind
            }

            return;
        }



        this.step++;

        this.step_rewards.Add(0);
        // add new entry in the rewards counting list
        // with this added 0 reward there is an entry for every step even if there was no reward signal encountered



        if (this.fixedTimesteps && this.episodeStatus == EpisodeStatus.WaitingForStep)
        {
            this.episodeStatus = EpisodeStatus.Running;
            this.stepFinished = false;
            this.timestepObstacleHit = false;

            this.timeOfLastStepBegin = Time.time;
        }

        if (this.fixedTimesteps == false)
        {
            this.timeOfLastStepBegin = Time.time;
            this.timestepObstacleHit = false;
        }
    }


    public void setCenterIndicators(List<GameObject> indicators)
    {
        this.centerIndicators = indicators;
        this.numberOfGoals = indicators.Count - 1;
        // there is a center indicator for the finishLine we thus need to substract one
    }

    public void StartEpisode(bool evalMode, CollisionMode collisionMode, MapType mt, string jetbot_name)
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
        this.obstacleHit = false;
        this.wallHit = false;
        this.hitObstacles = new List<GameObject>();
        this.mapType = mt;
        this.finishLineHit = false;

        this.timestepObstacleHit = false;

        this.evalMode = evalMode;
        this.collisionMode = collisionMode;

        this.jetbot_name = jetbot_name;

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
        topViewRecorder.StopVideo(this.duration);

        this.stepFinished = true;

    }

    public bool IsTerminated()
    {
        return (this.episodeStatus != EpisodeStatus.Running) && (this.episodeStatus != EpisodeStatus.WaitingForStep);
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
        info.Add("obstacleCollision", this.obstacleHit ? "1" : "0");
        info.Add("wallCollision", this.wallHit ? "1" : "0");

        info.Add("mapDifficulty", this.mapType.GetDifficulty());

        info.Add("jetbotType", this.jetbot_name);
        info.Add("finishLineHit", this.finishLineHit ? "1" : "0");

        return info;
    }

    public List<float> GetRewards()
    {

        return this.step_rewards;
    }

    public bool isEpisodeRunning()
    {
        if (this.fixedTimesteps)
        {
            return this.episodeStatus == EpisodeStatus.Running;
        }
        else
        {
            return this.episodeStatus == EpisodeStatus.Running || (this.episodeStatus == EpisodeStatus.WaitingForStep && this.step != -1);
        }
    }


    public void AddReward(float reward)
    {
        if (!isEpisodeRunning())
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

        float cosine_similarity = GetCosineSimilarityXZPlane(nextGoalDirection, agentOrientation);

        return cosine_similarity;
    }

    public static float GetCosineSimilarityXZPlane(Vector3 V1, Vector3 V2)
    {
        float result = (float)((V1.x * V2.x + V1.z * V2.z)
                / (Math.Sqrt(Math.Pow(V1.x, 2) + Math.Pow(V1.z, 2)) *
                    Math.Sqrt(Math.Pow(V2.x, 2) + Math.Pow(V2.z, 2))
                ));


        return result;
    }

    public void FixedUpdate()
    {
        if (this.duration >= this.allowedTime && this.episodeStatus == EpisodeStatus.Running)
        {
            AddEventReward(timeoutReward);
            if (this.numberOfGoals == this.passedGoals.Count)
            {
                EndEpisode(EpisodeStatus.Success);
            }
            else
            {
                EndEpisode(EpisodeStatus.Timeout);
            }
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

    public void IncreasePassedGoals(GameObject goalMiddle)
    {
        GameObject goal = goalMiddle.transform.parent.gameObject;
        int goalnumber = goal.transform.name[goal.transform.name.Length - 1] - '0';
        this.passedGoals.Add(goalnumber);
    }


    public void hitFinishLine(GameObject goal)
    {
        this.finishLineHit = true;

        if (this.passedGoals.Count == this.numberOfGoals)
        {
            AddEventReward(finishLineReward);


            EndEpisode(EpisodeStatus.Success);
        }
        else
        {
            AddEventReward(goalMissedReward);
            EndEpisode(EpisodeStatus.FinishWithoutAllGoals);
        }

    }

    public void hitWall(GameObject obstacle)
    {
        this.wallHit = true;
        handleCollision(obstacle, wallHitReward, EpisodeStatus.Collision);
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
            BallColor script = t.GetChild(i).GetComponent<BallColor>();
            if (script != null)
            {
                script.SetGreen();
            }
        }
    }

    public void colorRed(GameObject goal)
    {
        Transform t = goal.transform;
        for (int i = 0; i < t.childCount; i++)
        {
            BallColor script = t.GetChild(i).GetComponent<BallColor>();
            if (script != null)
            {
                script.SetRed();
            }
        }
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

    public void handleCollision(GameObject obstacle, float reward, EpisodeStatus potentialEndReason)
    {
        // returns true if the collision reward should be awarded

        this.obstacleOrWallHit = true;

        if (this.collisionMode == CollisionMode.oncePerEpisode)
        {
            if (this.hitObstacles.Contains(obstacle) == false)
            {
                this.hitObstacles.Add(obstacle);
                AddEventReward(reward);
            }
        }
        else if (this.collisionMode == CollisionMode.oncePerTimestep)
        {
            if (this.timestepObstacleHit == false)
            {
                this.timestepObstacleHit = true;
                AddEventReward(reward);
            }
            if (this.hitObstacles.Contains(obstacle) == false)
            {
                this.hitObstacles.Add(obstacle);
            }
        }
        else if (this.collisionMode == CollisionMode.terminate)
        {
            this.hitObstacles.Add(obstacle);

            AddEventReward(reward);
            if (this.evalMode == false)
            {
                EndEpisode(potentialEndReason);
            }
        }
        else if (this.collisionMode == CollisionMode.unrestricted)
        {
            AddEventReward(reward);
            if (this.hitObstacles.Contains(obstacle) == false)
            {
                this.hitObstacles.Add(obstacle);
            }
        }
        else if (this.collisionMode == CollisionMode.ignoreCollisions)
        {
            // we do not award the negative reward / penalty
            if (this.hitObstacles.Contains(obstacle) == false)
            {
                this.hitObstacles.Add(obstacle);
            }
        }
        else
        {
            Debug.LogError($"unknown collision mode {this.collisionMode}");
        }
    }

    public void redObstacleHit(GameObject obstacle)
    {
        obstacleHit = true;
        handleCollision(obstacle, obstacleHitReward, EpisodeStatus.Collision);
    }

    public void blueObstacleHit(GameObject obstacle)
    {
        obstacleHit = true;
        handleCollision(obstacle, obstacleHitReward, EpisodeStatus.Collision);
    }

    private void OnTriggerStay(Collider coll)
    {
        Collision(coll.gameObject);
    }

    private void OnTriggerEnter(Collider coll)
    {
        Collision(coll.gameObject);
    }

    private void OnCollisionStay(Collision coll)
    {
        Collision(coll.gameObject);
    }

    private void OnCollisionEnter(Collision coll)
    {
        Collision(coll.gameObject);
    }

    private void Collision(GameObject coll)
    {
        // This is attached to the JetBot

        if (!isEpisodeRunning())
        {
            //Debug.LogWarning("Episode not running, ignore collision with " + coll.tag);
            return;
        }
        if (coll.tag == "BlueObstacleTag")
        {
            blueObstacleHit(coll);
            return;
        }
        if (coll.tag == "RedObstacleTag")
        {
            redObstacleHit(coll);
            return;
        }
        if (coll.tag == "GoalPassed")
        {
            coll.tag = "Destroyed";
            goalPassed(coll);
            return;
        }
        if (coll.tag == "GoalMissed")
        {
            goalMissed(coll);
            return;
        }
        if (coll.tag == "Wall")
        {
            hitWall(coll);

            return;
        }
        if (coll.tag == "FinishLine")
        {
            hitFinishLine(coll);
            return;
        }
        if (coll.tag == "Destroyed")
        {
            // duplicate detection, ignore
            return;
        }
        if (coll.tag == "IgnoreCollision")
        {
            // ignore collision, e.g. the arena ground
            return;
        }
        Debug.LogError($"unknown tag {coll.tag} {coll.name}");
    }
}