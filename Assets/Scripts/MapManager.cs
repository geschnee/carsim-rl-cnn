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



    public Goal[] goals;

    public Vector3 jetBotSpawnPosition;
    public Quaternion jetBotSpawnRotation;

    public Vector3 finishCheckpointPosition;
}


public class MapManager : MonoBehaviour
{
    public GameObject allGoals;
    public GameObject finishLine;
    public GameObject JetBot;
    public List<UnityEngine.Object> obstacles = new List<UnityEngine.Object>();

    
	public List<GameObject> availibleJetbots;

    public Transform gameManagerTransform;
    public Vector3 gameManagerPosition;
    public GameObject obstacleBlue;
    public GameObject obstacleRed;
    public GameObject goalPassedGameObject;
    public GameObject goalMissedGameObject;
    public GameObject finishlineCheckpoint;
    public GameObject goalBall;

    private GameObject finishMissedGameObject;

    public GameObject goalMiddleIndicator;

    public List<GameObject> centerIndicators;

    // current map
	private MapData mapData;

    public void Awake(){
        this.gameManagerTransform = this.transform;
        this.gameManagerPosition = this.gameManagerTransform.position;
    }

    public MapData InitializeMapWithObstacles(MapType mapType, float jetBotRotation)
	{

		// generate a new map with new obstacle, decide which type of map should be generated
		mapData = this.GenerateObstacleMap(mapType, jetBotRotation);
		this.IntantiateObstacles(mapData);

		return mapData;
	}

    public GameObject GetJetBot(string jetBotName)
	{
		
		for (int i = 0; i < this.availibleJetbots.Count; i++)
		{
			if (this.availibleJetbots[i].name == jetBotName)
			{
				return this.availibleJetbots[i];
			}
		}
		Debug.LogError($"JetBot {jetBotName} not found, will use default JetBot {this.availibleJetbots[0].name}");
		return this.availibleJetbots[0];
	}

    public GameObject SpawnJetBot(MapData mapData, int instanceNumber, string jetbot_name)
    {

        Vector3 spawnPoint = mapData.jetBotSpawnPosition;

        Quaternion jbRotation = new Quaternion(0, 1, 0, 1); // OLD
        jbRotation = mapData.jetBotSpawnRotation;

        GameObject jetBot = GetJetBot(jetbot_name);

        GameObject jb = GameObject.Instantiate(original: jetBot, position: spawnPoint, rotation: jbRotation, this.gameManagerTransform.parent);
        jb.name = $"JetBot {instanceNumber}";

        jb.GetComponent<EpisodeManager>().setCenterIndicators(this.centerIndicators);

        return jb;
    }
    private Vector3 GetJetBotSpawnCoords()
    {

        int minXLocal = (int)(Constants.X_Map_MIN + this.gameManagerPosition.x);
        int z = (int)(this.gameManagerPosition.z + Constants.Z_WIDTH / 2);
        Vector3 SpawnPoint = new(minXLocal, Constants.JETBOT_SPAWN_Y, z);

        // in the middle of the short edge (z dimension)

        // -8, 0, 1 + x * 30

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

        Debug.LogError($"GetJetBotRandomCoordsEval called");


        return spawnPoint;
    }

    public Quaternion JetBotRotation(float jetBotSpawnRotationAngle)
    {
        // previously called JetBotRandomRotation
        // this method did generate the random angle, python-gymEnv now does that for better/easier control

        Quaternion originalQuaternion = new Quaternion(0, 1, 0, 1);

        // Convert to Euler angles
        Vector3 currentRotation = originalQuaternion.eulerAngles;

        // Add the random angle to the current rotation
        Vector3 modifiedRotation = currentRotation + new Vector3(0, jetBotSpawnRotationAngle, 0);

        // Convert back to quaternion
        Quaternion modifiedQuaternion = Quaternion.Euler(modifiedRotation);
        return modifiedQuaternion;
    }


    public void IntantiateObstacles(MapData goalList)
    {

        this.allGoals = new GameObject(name: "AllGoals");
        allGoals.transform.SetParent(this.gameManagerTransform.parent); // set goals to be child of TrainArena


        // instantiate all goals except the last one
        for (int i = 0; i < goalList.goals.Length - 1; i++)
        {

            Goal goal = goalList.goals[i];
            GameObject goalInstantiatedGameObject;


            goalInstantiatedGameObject = goal.InstantiateGoal(this.goalPassedGameObject, this.goalMissedGameObject, this.goalMiddleIndicator, this.gameManagerPosition, this.goalBall);
            goalInstantiatedGameObject.transform.SetParent(allGoals.transform);

            goalInstantiatedGameObject.name = "Goal" + i.ToString();

        }

        // initialize last goal with finish line checkpoint
        int lastGoalIndex = goalList.goals.Length - 1;
        Goal goalLast = goalList.goals[lastGoalIndex];
        GameObject goalInstantiatedGameObjectLast = goalLast.InstantiateGoal(this.goalPassedGameObject, this.goalMissedGameObject, this.goalMiddleIndicator, this.gameManagerPosition, this.goalBall);
        goalInstantiatedGameObjectLast.transform.SetParent(allGoals.transform);
        goalInstantiatedGameObjectLast.name = "Goal" + lastGoalIndex.ToString();


        // add finishLine after the last goal
        GameObject finishLine = InstantiateFinishLine();
        finishLine.transform.SetParent(this.gameManagerTransform.parent);
        this.finishLine = finishLine;

        this.centerIndicators = new List<GameObject>();

        for (int i = 0; i < allGoals.transform.childCount; i++)
        {
            GameObject goal = allGoals.transform.GetChild(i).gameObject;

            
            GameObject middle = FindChildWithTag(goal, "CenterIndicator");
            this.centerIndicators.Add(middle);
            if (middle == null)
            {
                Debug.LogWarning($"no child with tag CenterIndicator found, big error");
            }
            

        }
        if (this.centerIndicators.Count != goalList.goals.Length)
        {
            Debug.LogWarning($"only {this.centerIndicators.Count} center indicators found but there are {goalList.goals.Length} goals");
        }

        // finish after goal
        GameObject middleFinished = FindChildWithTag(finishLine, "CenterIndicator");
        this.centerIndicators.Add(middleFinished);
        if (middleFinished == null)
        {
            Debug.LogWarning($"no child with tag CenterIndicator found, big error");
        }

    }

    public GameObject InstantiateFinishLine(){
        GameObject finishParentGameObject = new(name: "FinishLine");
        
        Quaternion goalRotationQuaternion = new Quaternion(0, 0, 0, 0);

        Vector3 finishLinePosition = new Vector3(Constants.MAX_X + this.gameManagerPosition.x, Constants.SPAWNHEIGHT_Y, this.gameManagerPosition.z + Constants.Z_WIDTH / 2);

        GameObject finish = GameObject.Instantiate(this.finishlineCheckpoint, finishLinePosition, goalRotationQuaternion, finishParentGameObject.transform);

        
        GameObject.Instantiate(this.goalMiddleIndicator, finishLinePosition, goalRotationQuaternion, finishParentGameObject.transform);

        return finishParentGameObject;
    }


    public void DestroyMap()
    {
        Destroy(this.allGoals);
        Destroy(this.finishLine);
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


    // generate maps with different placement of obstacles
    public MapData GenerateObstacleMap(MapType mapType, float jetBotSpawnRotationAngle)
    {

        Vector3 jetBotSpawnPosition = this.GetJetBotSpawnCoords();


        Quaternion jetBotSpawnRotation = this.JetBotRotation(jetBotSpawnRotationAngle);


        Goal[] obstacles = new Goal[0];

        switch (mapType)
        {
            case MapType.easyBlueFirst:
                obstacles = this.GenerateEasyGoalLaneMiddleMap(true);
                break;
            case MapType.easyRedFirst:
                obstacles = this.GenerateEasyGoalLaneMiddleMap(false);
                break;
            case MapType.mediumBlueFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapMedium(true, true);
                break;
            case MapType.mediumBlueFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapMedium(true, false);
                break;
            case MapType.mediumRedFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapMedium(false, true);
                break;
            case MapType.mediumRedFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapMedium(false, false);
                break;


            case MapType.hardBlueFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapHard(true, true);
                break;
            case MapType.hardBlueFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapHard(true, false);
                break;
            case MapType.hardRedFirstLeft:
                obstacles = this.GenerateTwoGoalLanesMapHard(false, true);
                break;
            case MapType.hardRedFirstRight:
                obstacles = this.GenerateTwoGoalLanesMapHard(false, false);
                break;
        }

        MapData mapData = new MapData { goals = obstacles, jetBotSpawnPosition = jetBotSpawnPosition, jetBotSpawnRotation = jetBotSpawnRotation };

        return mapData;

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