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

	// Corner coordinates of the arena: (0, 0) (0, 10) (20,10) (20, 0)

	public int idOfCurrentRun;

	public GameObject JetBot;

	public List<GameObject> availibleJetbots;

	// need to load the prefabs of the obstacles in unity here
	public GameObject obstacleBlue;
	public GameObject obstacleRed;
	public GameObject goalPassedWallCheckpoint;
	public GameObject goalMissedWallCheckpoint;
	public GameObject FinishLineCheckpoint;
	public GameObject goalBall;

	public GameObject FinishLineMissedCheckpoint;

	public GameObject goalMiddleIndicator;

	// initialize obstacle Map Generator
	private ObstacleMapManager obstacleMapManager;

	// current map
	private MapData mapData;


	// Start is called before the first frame update
	void Start()
	{
		this.gameObject.AddComponent<ObstacleMapManager>();

		this.obstacleMapManager = this.gameObject.GetComponent<ObstacleMapManager>();

		this.obstacleMapManager.SetLikeInitialize(this.transform, obstacleBlue, obstacleRed, goalPassedWallCheckpoint, goalMissedWallCheckpoint, this.FinishLineCheckpoint, this.FinishLineMissedCheckpoint, goalMiddleIndicator, this.goalBall, this.JetBot);
	}

	public GameObject spawnJetbot(MapData md, int instanceNumber)
	{
		return this.obstacleMapManager.SpawnJetBot(md, instanceNumber);
	}


	public MapData InitializeMapWithObstacles(MapType mapType, int idOfCurrentRun, Spawn jetBotSpawn, float jetBotRotation)
	{

		// generate a new map with new obstacle, decide which type of map should be generated
		mapData = this.obstacleMapManager.GenerateObstacleMap(mapType, this.idOfCurrentRun, jetBotSpawn, jetBotRotation);
		this.obstacleMapManager.IntantiateObstacles(mapData);

		return mapData;
	}

	public void DestroyObstaclesOnMap()
	{
		this.obstacleMapManager.DestroyMap();
	}

	public void setJetbot(string jetbotName)
	{

		for (int i = 0; i < this.availibleJetbots.Count; i++)
		{
			if (this.availibleJetbots[i].name == jetbotName)
			{
				this.JetBot = this.availibleJetbots[i];
				return;
			}
		}
		Debug.LogError($"Jetbot {jetbotName} not found, will use default Jetbot {this.availibleJetbots[0].name}");
		this.JetBot = this.availibleJetbots[0];
	}
}