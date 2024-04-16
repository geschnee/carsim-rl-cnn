using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
//using Json.Net;
using UnityEngine;
using Random = System.Random;

static class Constants
{
    public const int X_Map_MIN = 3;
    public const int X_WIDTH_JETBOT = 3;

    public const int MIN_X = 7;
    public const int MAX_X = 18;
    public const int X_WIDTH = 14;

    //links und rechts vertauscht

    public const float LEFTMOST_Z = 10;
    public const float RIGHTMOST_Z = 0f;
    public const float Z_WIDTH = 10.5f;

    public const float SPAWNHEIGHT_Y = 1f;
    public const float MINWIDTH_GOAL = 2;
    public const float MAXWIDTH_GOAL = 4;
    public const int MINXDISTANCEGOALS = 4;
    public const int MAXXDISTANCEGOALS = 6;

    public const float JETBOT_SPAWN_Y = 0f;
}

public class Goal
{
    public GameObject ObstacleGO;
    public Vector3[] Coords;
    public Vector3 goalBallOffset = new Vector3(0, 1.5f, 0);

    public Goal(GameObject obstacle, Vector3[] coords)
    {
        ObstacleGO = obstacle;
        Coords = coords;
    }

    public GameObject InstantiateGoal(GameObject passedCheckpointWall, GameObject missedCheckpointWall, GameObject middleIndicator, Vector3 gameManagerPosition, GameObject goalBall)
    {
        Vector3 coords0 = this.Coords[0];
        Vector3 coords1 = this.Coords[1];
        float widthObstacle = this.ObstacleGO.transform.localScale.z;
        Quaternion goalRotationQuaternion = new Quaternion(0, 0, 0, 0);

        //local coordinates dependent from game manager position
        float localLeftMostZ = Constants.Z_WIDTH + gameManagerPosition.z;
        float localRightMostZ = gameManagerPosition.z;

        //parent Gameobject, add all single goal items to goalParentGameObject
        GameObject goalParentGameObject = new(name: "Goal");

        //goalposts
        GameObject.Instantiate(this.ObstacleGO, this.Coords[0], goalRotationQuaternion, goalParentGameObject.transform);
        GameObject.Instantiate(this.ObstacleGO, this.Coords[1], goalRotationQuaternion, goalParentGameObject.transform);

        //goalBalls
        GameObject.Instantiate(goalBall, this.Coords[0] + goalBallOffset, goalRotationQuaternion, goalParentGameObject.transform);
        GameObject.Instantiate(goalBall, this.Coords[1] + goalBallOffset, goalRotationQuaternion, goalParentGameObject.transform);


        // middleIndicator
        Vector3 midPoint = this.GetMidPoint(this.Coords[0], this.Coords[1]);
        GameObject.Instantiate(middleIndicator, midPoint, goalRotationQuaternion, goalParentGameObject.transform);

        //instantiate passed checkpoint wall between obstacles 
        Vector3 pos = coords1 - coords0;
        GameObject passedWall = GameObject.Instantiate(passedCheckpointWall, this.GetMidPoint(coords0, coords1), goalRotationQuaternion, goalParentGameObject.transform);

        Vector3 actScale = passedWall.transform.localScale;
        // calculate length between obstacles 
        passedWall.transform.localScale += new Vector3(0, 0, pos.magnitude - actScale.z - widthObstacle);

        //intantiate missed obstacle wall beside obstacles

        //left from right obstacles (bigger z is right when view from spawn)
        if (coords0.z > coords1.z)
        {

            // intatiate missed Checkpoint wall from left obstacle of the goal to the border   
            float distanceToZLeft = Math.Abs(localLeftMostZ - coords0.z);
            Vector3 pointLeftBorder = new Vector3(coords0.x, coords0.y, localLeftMostZ);
            Vector3 midPointToLeftBorder = this.GetMidPoint(coords0, pointLeftBorder);

            GameObject missedWallLeft = GameObject.Instantiate(missedCheckpointWall, midPointToLeftBorder, goalRotationQuaternion, goalParentGameObject.transform);
            missedWallLeft.transform.localScale += new Vector3(0, 0, distanceToZLeft - 1.25f * widthObstacle);


            // intatiate missed Checkpoint wall from right obstacle of the goal to the border   
            float distanceToZRight = Math.Abs(localRightMostZ - coords1.z);
            Vector3 pointRightBorder = new Vector3(coords1.x, coords1.y, localRightMostZ);
            Vector3 midPointToRightBorder = this.GetMidPoint(coords1, pointRightBorder);

            GameObject missedWallRight = GameObject.Instantiate(missedCheckpointWall, midPointToRightBorder, goalRotationQuaternion, goalParentGameObject.transform);
            missedWallRight.transform.localScale += new Vector3(0, 0, distanceToZRight - 1.5f * widthObstacle);


        }
        // other obstacle is more left (coord1 is more left then coord0)
        else
        {
            // intatiate missed Checkpoint wall from left obstacle of the goal to the border   
            float distanceToZLeft = Math.Abs(localLeftMostZ - coords1.z);
            Vector3 pointLeftBorder = new Vector3(coords1.x, coords1.y, localLeftMostZ);
            Vector3 midPointToLeftBorder = this.GetMidPoint(coords1, pointLeftBorder);

            GameObject missedWallLeft = GameObject.Instantiate(missedCheckpointWall, midPointToLeftBorder, goalRotationQuaternion, goalParentGameObject.transform);
            missedWallLeft.transform.localScale += new Vector3(0, 0, distanceToZLeft - 1.25f * widthObstacle);

            // intatiate missed Checkpoint wall from right obstacle of the goal to the border   
            float distanceToZRight = Math.Abs(localRightMostZ - coords0.z);
            Vector3 pointRightBorder = new Vector3(coords0.x, coords0.y, localRightMostZ);
            Vector3 midPointToRightBorder = this.GetMidPoint(coords0, pointRightBorder);

            GameObject missedWallRight = GameObject.Instantiate(missedCheckpointWall, midPointToRightBorder, goalRotationQuaternion, goalParentGameObject.transform);
            missedWallRight.transform.localScale += new Vector3(0, 0, distanceToZRight - 1.5f * widthObstacle);
        }

        return goalParentGameObject;
    }

    private Vector3 GetMidPoint(Vector3 a, Vector3 b)
    {
        Vector3 c = a + b;
        Vector3 midPoint = new Vector3(0.5f * c.x, 0.5f * c.y, 0.5f * c.z);
        return midPoint;
    }

}

public class MapData
{
    // Maximilian called this class ObstacleList
    // it was simply a list of goals

    // now also the spawn position of the JetBot is saved in this class
    // including the spawn rotation of course
    // makes for more complete serialization and cleaner code since the spawn position is closely related to the goals
    // finishCheckpoint should also be spawned behind the last goal
    // else there will be premature success
    // there should be no collisions with the last obstacle after the last goal is completed


    public int listId;
    public Goal[] goals;

    public Vector3 jetBotSpawnPosition;
    public Quaternion jetBotSpawnRotation;

    public Vector3 finishCheckpointPosition;
}


public class ObstacleMapManager : MonoBehaviour
{
    public List<UnityEngine.Object> obstacles = new List<UnityEngine.Object>();

    private Boolean isFinishLineLastGoal;


    public Transform gameManagerTransform;
    public Vector3 gameManagerPosition;
    public GameObject obstacleBlue;
    public GameObject obstacleRed;
    public GameObject goalPassedGameOjbect;
    public GameObject goalMissedGameObject;
    public GameObject finishlineCheckpoint;
    public GameObject goalBall;

    private GameObject finishMissedGameObject;

    public GameObject goalMiddleIndicator;

    public GameObject allGoals;
    public GameObject JetBot;
    public double JetBotXSpawn;

    private List<GameObject> centerIndicators;

    public ObstacleMapManager(Transform gameManagerTransform, GameObject obstacleBlue, GameObject obstacleRed, GameObject goalPassedGameObject, GameObject goalMissedGameObject, GameObject finishlineCheckpoint, GameObject goalBall, Boolean isFinishLine, GameObject JetBot)
    {
        Debug.LogWarning("ObstacleMapManager constructor called, this is unexpected");
    }

    public void SetLikeInitialize(Transform gameManagerTransform, GameObject obstacleBlue, GameObject obstacleRed, GameObject goalPassedGameObject, GameObject goalMissedGameObject, GameObject finishlineCheckpoint, GameObject FinishLineMissedCheckpoint, GameObject goalMiddleIndicator, GameObject goalBall, Boolean isFinishLine, GameObject JetBot)
    {
        this.gameManagerTransform = gameManagerTransform;
        this.gameManagerPosition = gameManagerTransform.position;
        this.obstacleBlue = obstacleBlue;
        this.obstacleRed = obstacleRed;
        this.goalPassedGameOjbect = goalPassedGameObject;
        this.goalMissedGameObject = goalMissedGameObject;
        this.finishlineCheckpoint = finishlineCheckpoint;
        this.finishMissedGameObject = FinishLineMissedCheckpoint;
        this.goalMiddleIndicator = goalMiddleIndicator;
        this.goalBall = goalBall;

        this.isFinishLineLastGoal = isFinishLine;
        this.JetBot = JetBot;

    }

    public GameObject SpawnJetBot(MapData mapData, int instanceNumber)
    {

        Vector3 spawnPoint = mapData.jetBotSpawnPosition;

        Quaternion jbRotation = new Quaternion(0, 1, 0, 1); // OLD
        jbRotation = mapData.jetBotSpawnRotation;

        GameObject jb = GameObject.Instantiate(original: this.JetBot, position: spawnPoint, rotation: jbRotation, this.gameManagerTransform.parent);
        jb.name = $"JetBot {instanceNumber}";

        jb.GetComponent<EpisodeManager>().setCenterIndicators(this.centerIndicators);

        //Debug.Log($"Jetbot spawn rotation y {jb.transform.rotation.y}");
        return jb;
    }
    private Vector3 GetJetBotSpawnCoords()
    {

        int minXLocal = (int)(Constants.X_Map_MIN + this.gameManagerPosition.x);
        int z = (int)(this.gameManagerPosition.z + Constants.Z_WIDTH / 2);
        Vector3 SpawnPoint = new(minXLocal, Constants.JETBOT_SPAWN_Y, z);

        // in the middle of the short edge (z dimension)

        return SpawnPoint;
    }

    private Vector3 GetJetBotRandomCoords()
    {
        //local goal post coordinates depend arena position
        float zLeftMax = (2 + this.gameManagerPosition.z);
        // left post of goal max 
        float zRightMax = (this.gameManagerPosition.z + Constants.Z_WIDTH - 2);
        int minXLocal = (int)(Constants.X_Map_MIN + this.gameManagerPosition.x);
        int maxXLocal = minXLocal + Constants.X_WIDTH - Constants.MINXDISTANCEGOALS;

        Random rnd = new Random();
        float zRandomCoord = (float)rnd.NextDouble() * (zRightMax - zLeftMax) + zLeftMax;
        float xRandomCoord = (float)rnd.NextDouble() * (maxXLocal - minXLocal) + minXLocal;
        Vector3 spawnPoint = new(xRandomCoord, Constants.JETBOT_SPAWN_Y, zRandomCoord);
        //this.JetBotXSpawn = xRandomCoord;


        //Debug.Log($"JetBot spawn range x {minXLocal} - {maxXLocal} z {zLeftMax} - {zRightMax}");

        // TODO vielleicht muss die X Position angepasst werden, da die eigene Position des JetBots nun der Position des coarBodys entspricht
        // das war zuvor nicht der Fall und hat zu stark unterschiedlichen Positionen bei Rotation gefuehrt
        // PythonJetbot centered prefab

        return spawnPoint;
    }

    private Vector3 GetJetBotRandomCoordsEval()
    {
        // always spawn on the fixed x line

        //local goal post coordinates depend arena position
        float zLeftMax = (2 + this.gameManagerPosition.z);
        // left post of goal max 
        float zRightMax = (this.gameManagerPosition.z + Constants.Z_WIDTH - 2);
        int minXLocal = (int)(Constants.X_Map_MIN + this.gameManagerPosition.x);

        Random rnd = new Random();
        float zRandomCoord = (float)rnd.NextDouble() * (zRightMax - zLeftMax) + zLeftMax;
        Vector3 spawnPoint = new(minXLocal, Constants.JETBOT_SPAWN_Y, zRandomCoord);


        //Debug.LogWarning($"Eval JetBot spawn x {minXLocal} - range z {zLeftMax} - {zRightMax}");

        return spawnPoint;
    }

    // TODO was this method used in Maximilian code before?
    // yes, in CarAgent random spawn
    public Quaternion JetBotRandomRotation(Boolean veryRandom)
    {
        Quaternion originalQuaternion = new Quaternion(0, 1, 0, 1);

        // Convert to Euler angles
        Vector3 currentRotation = originalQuaternion.eulerAngles;

        float randomAngle;
        if (veryRandom) {
            // Generate a random angle between -45 and 45 degrees
            randomAngle = UnityEngine.Random.Range(-45f, 45f);
        } else {
            randomAngle = UnityEngine.Random.Range(-15f, 15f);
        }

        // Add the random angle to the current rotation
        Vector3 modifiedRotation = currentRotation + new Vector3(0, randomAngle, 0);

        // Convert back to quaternion
        Quaternion modifiedQuaternion = Quaternion.Euler(modifiedRotation);
        return modifiedQuaternion;
    }


    public void IntantiateObstacles(MapData goalList)
    {

        this.allGoals = new GameObject(name: "AllGoals");
        allGoals.transform.SetParent(this.gameManagerTransform.parent); // set goals to be child of TrainArena

        // TODO the finish line is always placed directly on the last goal
        // maybe the finish line should be a bit behind it.

        // instantiate all goals except the last one
        for (int i = 0; i < goalList.goals.Length - 1; i++)
        {

            Goal goal = goalList.goals[i];
            GameObject goalInstantiatedGameObject;
            

            goalInstantiatedGameObject = goal.InstantiateGoal(this.goalPassedGameOjbect, this.goalMissedGameObject, this.goalMiddleIndicator, this.gameManagerPosition, this.goalBall);
            goalInstantiatedGameObject.transform.SetParent(allGoals.transform);

            goalInstantiatedGameObject.name = "Goal" + i.ToString();

        }

        // initialize last goal with finish line checkpoint
        int lastGoalIndex = goalList.goals.Length - 1;
        Goal goalLast = goalList.goals[lastGoalIndex];
        GameObject goalInstantiatedGameObjectLast = goalLast.InstantiateGoal(this.finishlineCheckpoint, this.finishMissedGameObject, this.goalMiddleIndicator, this.gameManagerPosition, this.goalBall);
        goalInstantiatedGameObjectLast.transform.SetParent(allGoals.transform);
        goalInstantiatedGameObjectLast.name = "Goal" + lastGoalIndex.ToString();


        this.centerIndicators = new List<GameObject>();
        //Debug.Log($"allGoals {allGoals}");
        //Debug.Log($"allGoals child amount {allGoals.transform.childCount}");

        this.centerIndicators = new List<GameObject>();

        for (int i = 0; i < allGoals.transform.childCount; i++)
        {
            GameObject goal = allGoals.transform.GetChild(i).gameObject;
            //Debug.Log($"goal {goal}");

            if (i == allGoals.transform.childCount - 1)
            {
                GameObject middleFinished = FindChildWithTag(goal, "FinishCheckpoint");
                this.centerIndicators.Add(middleFinished);
                if (middleFinished == null)
                {
                    Debug.LogWarning($"no child with tag FinishCheckpoint found, big error");
                }
            }
            else
            {
                GameObject middle = FindChildWithTag(goal, "GoalPassed");
                this.centerIndicators.Add(middle);
                if (middle == null)
                {
                    Debug.LogWarning($"no child with tag GoalPassed found, big error");
                }
            }

        }
        if (this.centerIndicators.Count != goalList.goals.Length)
        {
            Debug.LogWarning($"only {this.centerIndicators.Count} center indicators found but there are {goalList.goals.Length} goals");
        }

    }


    public void DestroyMap()
    {
        Destroy(this.allGoals);
    }

    public void DestroyObstacles()
    {
        for (int i = 0; i < this.obstacles.Count; i++)
        {
            GameObject.DestroyImmediate(this.obstacles[i]);
        }
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
        //Debug.LogWarning($"no child with tag {tag} found");
        return null;
    }

    public MapData LoadObstacleMap(string filepath, float id)
    {
        string fullPath = filepath + id.ToString() + ".json";
        if (File.Exists(fullPath))
        {
            string content = File.ReadAllText(fullPath);
            MapData mapData = JsonUtility.FromJson<MapData>(content);
            return mapData;

        }
        else
            throw new FileNotFoundException(
                "File not found.");
    }

    public void SaveObstacleMap(string filepath, float id, MapData mapData)
    {
        string fullPath = filepath + id.ToString() + ".json";
        string json = JsonUtility.ToJson(mapData);
        File.WriteAllText(fullPath, json);
    }


    // generate maps with different placement of obstacles
    public MapData GenerateObstacleMap(MapType mapType, int id, Spawn jetBotSpawn)
    {


        Vector3 jetBotSpawnPosition;
        if (jetBotSpawn == Spawn.Fixed || jetBotSpawn == Spawn.OrientationRandom || jetBotSpawn == Spawn.OrientationVeryRandom)
        {
            jetBotSpawnPosition = this.GetJetBotSpawnCoords();

        }
        else if (mapType == MapType.random)
        {
            jetBotSpawnPosition = this.GetJetBotRandomCoords();
            //Debug.Log($"Random JetBot spawn position {jetBotSpawnPosition}");
        }
        else
        {
            // the mapType is not random (it is an evaluation track)
            // we cannot use the same random spawn for these
            jetBotSpawnPosition = this.GetJetBotRandomCoordsEval();
        }

        Quaternion jetBotSpawnRotation;
        if (jetBotSpawn == Spawn.OrientationVeryRandom | jetBotSpawn == Spawn.FullyRandom)
        {
            jetBotSpawnRotation = this.JetBotRandomRotation(true);
        } else if (jetBotSpawn == Spawn.OrientationRandom)
        {
            jetBotSpawnRotation = this.JetBotRandomRotation(false);
        }
        else
        {
            jetBotSpawnRotation = new Quaternion(0, 1, 0, 1);
        }
        Debug.Log($"JetBot spawn rotation y {jetBotSpawnRotation.eulerAngles.y}");

        Goal[] obstacles = new Goal[0];

        switch (mapType)
        {
            case MapType.random:
                obstacles = this.GenerateRandomObstacleMap(jetBotSpawn, jetBotSpawnPosition);
                //Debug.Log("Random Map generated");
                break;
            case MapType.easyBlueFirst:
                obstacles = this.GenerateEasyGoalLaneMiddleMap(true);
                //Debug.Log("Easy middle lane with blue obstacles first map generated");
                break;
            case MapType.easyRedFirst:
                obstacles = this.GenerateEasyGoalLaneMiddleMap(false);
                //Debug.Log("Easy middle lane with red obstacles first map generated");
                break;
            case MapType.mediumBlueFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapMedium(true, true);
                //Debug.Log("Two lanes map with blue obstacles first generated");
                break;
            case MapType.mediumBlueFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapMedium(true, false);
                //Debug.Log("Two lanes map with blue obstacles first generated");
                break;
            case MapType.mediumRedFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapMedium(false, true);
                //Debug.Log("Two lanes map with red obstacles first generated");
                break;
            case MapType.mediumRedFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapMedium(false, false);
                //Debug.Log("Two lanes map with red obstacles first generated");
                break;


            case MapType.hardBlueFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapHard(true, true);
                //Debug.Log("Two lanes map with blue obstacles first generated");
                break;
            case MapType.hardBlueFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapHard(true, false);
                //Debug.Log("Two lanes map with blue obstacles first generated");
                break;
            case MapType.hardRedFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapHard(false, true);
                //Debug.Log("Two lanes map with red obstacles first generated");
                break;
            case MapType.hardRedFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapHard(false, false);
                //Debug.Log("Two lanes map with red obstacles first generated");
                break;
        }

        MapData mapData = new MapData { listId = id, goals = obstacles, jetBotSpawnPosition = jetBotSpawnPosition, jetBotSpawnRotation = jetBotSpawnRotation };

        return mapData;

    }

    private Goal[] GenerateRandomObstacleMap(Spawn jetBotSpawn, Vector3 jetBotSpawnPosition)
    {

        List<Goal> obstacles = new List<Goal>();
        Random rnd = new Random();
        bool randomIsBlueFirst = rnd.Next(0, 2) == 0;

        GameObject actualColorObject = getColorObject(randomIsBlueFirst);

        //local goal post coordinates depend arena position
        float zLeftMax = (1 + this.gameManagerPosition.z);
        // left post of goal max 
        float zRightMax = (5.5f + this.gameManagerPosition.z);
        int minXLocal = (int)(Constants.MIN_X + this.gameManagerPosition.x);
        int maxXLocal = minXLocal + Constants.X_WIDTH;

        if (jetBotSpawn == Spawn.FullyRandom)
        {
            //first goal random distance to JetBot
            minXLocal = (int)(jetBotSpawnPosition.x + rnd.Next(Constants.MINXDISTANCEGOALS, Constants.MAXXDISTANCEGOALS));
            //Debug.Log($"minXLocal {minXLocal} maxXLocal {maxXLocal}");
        }

        // choose random distance between goals every round
        int xDistanceGoals = 0;
        for (int x = minXLocal; x < maxXLocal; x += xDistanceGoals)
        {
            // choose distance to next goal
            xDistanceGoals = rnd.Next(Constants.MINXDISTANCEGOALS, Constants.MAXXDISTANCEGOALS + 1);
            // choose z-position of left goal post random 
            float zLeftPost = (float)rnd.NextDouble() * (zRightMax - zLeftMax) + zLeftMax;

            Vector3[] coordsGoal = { new Vector3(), new Vector3() };

            Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftPost);

            Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftPost + Constants.MAXWIDTH_GOAL);

            coordsGoal[0] = coordLeft;
            coordsGoal[1] = coordRight;


            Goal goal = new Goal(actualColorObject, coordsGoal);
            obstacles.Add(goal);

            actualColorObject = actualColorObject == obstacleBlue ? obstacleRed : obstacleBlue;

        }

        return obstacles.ToArray();
    }

    private Goal[] GenerateEasyGoalLaneMiddleMap(Boolean isBlueFirst = true)
    {
        List<Goal> obstacles = new List<Goal>();
        // calculate how many goals


        // local coordinates fore every train arena
        float zLeftRow = (Constants.Z_WIDTH / 2 + this.gameManagerPosition.z - Constants.MAXWIDTH_GOAL / 2);

        int minXLocal = (int)(Constants.MIN_X + this.gameManagerPosition.x);
        int maxXLocal = minXLocal + Constants.X_WIDTH - 2;
        GameObject actualColorObject = getColorObject(isBlueFirst);


        for (int x = minXLocal; x < maxXLocal; x += Constants.MINXDISTANCEGOALS + 1)
        {

            // left obstacles of goals
            Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow);

            Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow + Constants.MAXWIDTH_GOAL);
            Vector3[] coordsGoal = { coordLeft, coordRight };

            Goal goal = new Goal(actualColorObject, coordsGoal);
            obstacles.Add(goal);

            actualColorObject = actualColorObject == obstacleBlue ? obstacleRed : obstacleBlue;

        }
        return obstacles.ToArray();
    }

    private Goal[] GenerateTwoGoalLanesMapHard(Boolean isBlueFirst = true, Boolean isLeftFirst = true)
    {

        List<Goal> obstacles = new List<Goal>();

        GameObject actualColorObject = getColorObject(isBlueFirst);

        bool left = isLeftFirst;

        //local goal post coordinaents depend arena position
        float zLeftRow = (1.5f + this.gameManagerPosition.z);
        float zRightRow = (5f + this.gameManagerPosition.z);
        int minXLocal = (int)(Constants.MIN_X + this.gameManagerPosition.x);
        int maxXLocal = minXLocal + Constants.X_WIDTH - 2;

        for (int x = minXLocal; x < maxXLocal; x += Constants.MAXXDISTANCEGOALS - 1)
        {
            Vector3[] coordsGoal = { new Vector3(), new Vector3() };

            //goal on left side
            if (left)
            {
                Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow);

                Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow + Constants.MAXWIDTH_GOAL);

                coordsGoal[0] = coordLeft;
                coordsGoal[1] = coordRight;

            }
            else
            {
                Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zRightRow);

                Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zRightRow + Constants.MAXWIDTH_GOAL);

                coordsGoal[0] = coordLeft;
                coordsGoal[1] = coordRight;
            }

            Goal goal = new Goal(actualColorObject, coordsGoal);
            obstacles.Add(goal);

            left = left == true ? false : true;
            actualColorObject = actualColorObject == obstacleBlue ? obstacleRed : obstacleBlue;
        }
        return obstacles.ToArray();
    }

    private Goal[] GenerateTwoGoalLanesMapMedium(Boolean isBlueFirst = true, Boolean isLeftFirst = true)
    {

        List<Goal> obstacles = new List<Goal>();

        GameObject actualColorObject = getColorObject(isBlueFirst);

        bool left = isLeftFirst;

        //local goal post coordinaents depend arena position
        float zLeftRow = (2.5f + this.gameManagerPosition.z);
        float zRightRow = (4f + this.gameManagerPosition.z);
        int minXLocal = (int)(Constants.MIN_X + this.gameManagerPosition.x);
        int maxXLocal = minXLocal + Constants.X_WIDTH - 2;


        for (int x = minXLocal; x < maxXLocal; x += Constants.MAXXDISTANCEGOALS - 1)
        {
            Vector3[] coordsGoal = { new Vector3(), new Vector3() };

            //goal on left side
            if (left)
            {
                Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zRightRow);

                Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zRightRow + Constants.MAXWIDTH_GOAL);

                coordsGoal[0] = coordLeft;
                coordsGoal[1] = coordRight;

            }
            else
            {
                Vector3 coordLeft = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow);

                Vector3 coordRight = new Vector3(x, Constants.SPAWNHEIGHT_Y, zLeftRow + Constants.MAXWIDTH_GOAL);

                coordsGoal[0] = coordLeft;
                coordsGoal[1] = coordRight;

            }

            Goal goal = new Goal(actualColorObject, coordsGoal);
            obstacles.Add(goal);

            left = left == true ? false : true;
            actualColorObject = actualColorObject == obstacleBlue ? obstacleRed : obstacleBlue;
        }

        return obstacles.ToArray();
    }

    private GameObject getColorObject(bool isBlueFirst)
    {
        if (isBlueFirst)
        {
            return this.obstacleBlue;
        }
        else
        {
            return this.obstacleRed;
        }
    }
}