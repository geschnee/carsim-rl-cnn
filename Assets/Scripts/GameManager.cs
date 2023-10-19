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
	 *This class controls the Game settings. In Unity & in this class you can choose which type pf map shoudl be
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
	private ObstacleList obstacleList;

	// generate map
	//could be selected in unity in the GameManager game object 
	public MapType mapTypeGeneratedMap;
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
		// load obstacles
		this.isEvaluation = !isTrainingSpawnRandom;
		if (this.isTrainingSpawnRandom == false)
		{
			isTrainingSpawnRandom = false;
			isLogTraining = false;


		}
		this.gameObject.AddComponent<ObstacleMapManager>();

		this.obstacleMapManager = this.gameObject.GetComponent<ObstacleMapManager>();
		//this.obstacleMapManager.gameManagerTransform = this.transform;
		this.obstacleMapManager.SetLikeInitialize(this.transform, obstacleBlue, obstacleRed, goalPassedWallCheckpoint, goalMissedWallCheckpoint, this.FinishLineCheckpoint, goalMiddleIndicator, this.isFinishLineLastGoal, this.JetBot, this.isTrainingSpawnRandom, this.singleGoalTraining);


		// this.obstacleMapManager = new ObstacleMapManager(this.transform, obstacleBlue, obstacleRed, goalPassedWallCheckpoint, goalMissedWallCheckpoint, this.FinishLineCheckpoint, this.isFinishLineLastGoal, this.JetBot, this.isTrainingSpawnRandom, this.singleGoalTraining);
		//this.obstacleMapManager.SpawnJetBot();



		InitializeMapWithObstacles(true);
	}

	public GameObject spawnJetbot()
	{

		print("before spawn jetbot in GameManager");
		return this.obstacleMapManager.SpawnJetBot();
	}

	void FixedUpdate()//FixedUpdate is called at a constant interval
	{

	}

	// Update is called once per frame
	void Update()
	{

	}


	public void InitializeMapWithObstacles(Boolean correctStart)
	{

		// load a already generated map
		if (loadObstacles)
		{
			obstacleList = this.obstacleMapManager.LoadObastacleMap(this.loadObstacleMapFilePath, this.idOfCurrentRun);
		}
		else if (!isEvaluation)
		{

			// generate a new map with new obstacle, decide which type of map should be generated
			obstacleList = this.obstacleMapManager.GenerateObstacleMap(this.mapTypeGeneratedMap, this.idOfCurrentRun);
			this.obstacleMapManager.IntantiateObstacles(obstacleList);


			if (this.saveObstacles)
			{
				this.obstacleMapManager.SaveObstacleMap(this.saveObstacleMapFilePath,
					this.idOfCurrentRun, obstacleList);

			}
		}

		else
		{

			if (currentMapIndex == evaluationMaps.Length - 1 && idOfCurrentRun == numberOfRunsPerMap - 1 && isEvaluation)
			{
				UnityEditor.EditorApplication.isPlaying = false;
				UnityEditor.EditorApplication.ExitPlaymode();

				Application.Quit();


			}
			else
			{
				this.mapTypeGeneratedMap = this.evaluationMaps[currentMapIndex];
				obstacleList = this.obstacleMapManager.GenerateObstacleMap(this.mapTypeGeneratedMap, this.idOfCurrentRun);
				// intantiate real objects in unity
				this.obstacleMapManager.IntantiateObstacles(obstacleList);
				idOfCurrentRun++;

				if (idOfCurrentRun == numberOfRunsPerMap)
				{
					currentMapIndex++;
					idOfCurrentRun = 0;

				}

			}
		}



	}

	//retuns coords at beginning of the map (start point)
	public Vector3 GetStartSpawnPosition()
	{
		return this.obstacleMapManager.GetJetBotSpawnCoords();
	}

	//returns random spawn position on map
	public Vector3 GetRandomSpawnPosition()
	{
		return this.obstacleMapManager.GetJetBotRandomCoords();

	}

	public Quaternion GetRandomSpawnRotation()
	{
		return this.obstacleMapManager.JetBotRandomRotation();
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

	public String GetMapTypeName()
	{
		MapType[] mapTypes = (MapType[])Enum.GetValues(typeof(MapType));

		return mapTypes[(int)this.mapTypeGeneratedMap].ToString();

	}

}
