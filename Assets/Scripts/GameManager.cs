using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class GameManager : MonoBehaviour
{
	/*
	 *This class controls the Game settings. In Unity & in this class you can choose which type of map should be
	 *generated and if there car should be controlled by a human or AI.
	 */


	// Corner coordinates of the arena: (0, 0) (0, 10) (20,10) (20, 0)


	public int idOfCurrentRun;

	public GameObject JetBot;

	// need to load the prefabs of the obstacles in unity here
	public GameObject obstacleBlue;
	public GameObject obstacleRed;
	public GameObject goalPassedWallCheckpoint;
	public GameObject goalMissedWallCheckpoint;
	public GameObject FinishLineCheckpoint;

	public GameObject goalMiddleIndicator;

	//spawn jetbot random on map if trainnig
	public Boolean isTrainingSpawnRandom;
	public bool singleGoalTraining;

	// has the last goal the finish line?
	public Boolean isFinishLineLastGoal = true;

	// initialize obstacle Map Generator
	private ObstacleMapManager obstacleMapManager;

	// current map
	private MapData mapData;

	public MapType[] evaluationMaps = new MapType[] {
		//MapType.easyGoalLaneMiddleBlueFirst,
		//MapType.easyGoalLaneMiddleRedFirst,

		MapType.twoGoalLanesBlueFirstLeftMedium,
		MapType.twoGoalLanesBlueFirstRightMedium,
		MapType.twoGoalLanesRedFirstLeftMedium,
		MapType.twoGoalLanesRedFirstRightMedium,

		//MapType.twoGoalLanesBlueFirstLeftHard,
		//MapType.twoGoalLanesBlueFirstRightHard,
		//MapType.twoGoalLanesRedFirstLeftHard,
		//MapType.twoGoalLanesRedFirstRightHard
	};

	// load obstacle Map
	public bool loadObstacles = false;
	public string loadObstacleMapFilePath = ".";

	// save map if generated
	public bool saveObstacles = false;
	public string saveObstacleMapFilePath = ".";


	//to store result
	private int result;

	//  
	// log in training
	public Boolean isLogTraining;
	public string resultsPath = "./results/results.csv";

	public Boolean isEvaluation;
	public int numberOfRunsPerMap = 10;
	private int currentMapIndex = 0;

	// Start is called before the first frame update
	void Start()
	{

		if (this.isTrainingSpawnRandom == false)
		{
			isTrainingSpawnRandom = false;
			isLogTraining = false;
		}
		this.gameObject.AddComponent<ObstacleMapManager>();

		this.obstacleMapManager = this.gameObject.GetComponent<ObstacleMapManager>();
		//this.obstacleMapManager.gameManagerTransform = this.transform;
		this.obstacleMapManager.SetLikeInitialize(this.transform, obstacleBlue, obstacleRed, goalPassedWallCheckpoint, goalMissedWallCheckpoint, this.FinishLineCheckpoint, goalMiddleIndicator, this.isFinishLineLastGoal, this.JetBot);

	}

	public GameObject spawnJetbot(MapData md, int instanceNumber)
	{
		return this.obstacleMapManager.SpawnJetBot(md, instanceNumber);
	}

	public void InitializeMapWithObstaclesFromFile(string loadObstacleMapFilePath, int idOfCurrentRun)
	{

		// load a already generated map

		mapData = this.obstacleMapManager.LoadObstacleMap(this.loadObstacleMapFilePath, this.idOfCurrentRun);

		// TODO why is there no:
		// this.obstacleMapManager.IntantiateObstacles(obstacleList);

	}

	public MapData InitializeMapWithObstacles(MapType currentMapIndex, int idOfCurrentRun, bool jetBotSpawnpointRandom, bool singleGoalTraining)
	{
		// TODO rewrite to use the passed parameters
		// I do not want magic in this function/class here

		//Debug.Log($"InitializeMapWithObstacles() called, currentMapIndex: {currentMapIndex}, idOfCurrentRun: {idOfCurrentRun}");

		MapType mapType = currentMapIndex;

		// generate a new map with new obstacle, decide which type of map should be generated
		mapData = this.obstacleMapManager.GenerateObstacleMap(mapType, this.idOfCurrentRun, jetBotSpawnpointRandom, singleGoalTraining);
		this.obstacleMapManager.IntantiateObstacles(mapData);


		if (this.saveObstacles)
		{
			this.obstacleMapManager.SaveObstacleMap(this.saveObstacleMapFilePath,
				this.idOfCurrentRun, mapData);
		}

		return mapData;
	}

	public void DestroyObstaclesOnMap()
	{
		this.obstacleMapManager.DestroyMap();
	}
	public Boolean GetIsTrainingSpawnRandom()
	{
		return this.isTrainingSpawnRandom;

	}
	public int GetIdOfCurrentRun()
	{
		return this.idOfCurrentRun;
	}

	public String GetMapTypeName(MapType mt)
	{
		return mt.ToString();
	}
}
