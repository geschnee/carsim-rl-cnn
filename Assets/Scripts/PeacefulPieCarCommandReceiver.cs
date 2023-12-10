using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;

using System.Text;
using System;

using System.IO;
using System.Linq;


public class StepReturnObject
{
    public string observation;
    public float reward;
    public bool done;
    public bool terminated;

    public Dictionary<string, string> info;

    public List<float> bootstrapped_rewards;

    public StepReturnObject(string observation, float reward, bool done, bool terminated, Dictionary<string, string> info, List<float> bootstrapped_rewards)
    {
        this.observation = observation;
        this.reward = reward;
        this.done = done;
        this.terminated = terminated;
        this.info = info;
        this.bootstrapped_rewards = bootstrapped_rewards;
    }
}

public class PeacefulPieCarCommandReceiver : MonoBehaviour
{
    class Rpc : JsonRpcService
    {

        GameObject arenaPrefab;

        private List<Arena> arenas = new List<Arena>();

        private Vector3 arenaPosition = new Vector3(0, 0, 0);
        private Vector3 arenaOffset;


        public Rpc(GameObject arenaPrefab, Vector3 arenaOffset)
        {
            this.arenaPrefab = arenaPrefab;
            this.arenaOffset = arenaOffset;
        }

        [JsonRpcMethod]
        void destroyMap(int id)
        {
            arenas[id].destroyMap();
        }

        [JsonRpcMethod]
        void deleteAllArenas()
        {
            foreach (Arena arena in arenas)
            {
                Destroy(arena.gameObject);
            }
            arenas.Clear();
        }


        [JsonRpcMethod]
        string reset(int id, string mapType, bool spawnpointRandom, bool singleGoalTraining, int bootstrap_n, float lightMultiplier)
        {
            //Debug.Log($"mapType: {mapType}");
            MapType mt = (MapType)Enum.Parse(typeof(MapType), mapType);
            //Debug.Log($"mt: {mt}");
            return arenas[id].reset(mt, spawnpointRandom, singleGoalTraining, bootstrap_n, lightMultiplier);
        }


        [JsonRpcMethod]
        void say(string message)
        {
            Debug.Log($"you sent {message}");
        }

        [JsonRpcMethod]
        void forwardInputsToCar(int id, float inputAccelerationLeft, float inputAccelerationRight)
        {
            arenas[id].forwardInputsToCar(inputAccelerationLeft, inputAccelerationRight);
        }

        [JsonRpcMethod]
        StepReturnObject immediateStep(int id, float inputAccelerationLeft, float inputAccelerationRight)
        {
            return arenas[id].immediateStep(inputAccelerationLeft, inputAccelerationRight);
        }

        [JsonRpcMethod]
        void asyncStepPart1(int id, float inputAccelerationLeft, float inputAccelerationRight)
        {
            arenas[id].asyncStepPart1(inputAccelerationLeft, inputAccelerationRight);
        }

        [JsonRpcMethod]
        StepReturnObject asyncStepPart2(int id)
        {
            return arenas[id].asyncStepPart2();
        }

        [JsonRpcMethod]
        void startArena(int id)
        {
            // spawn a new arena including gameManager and everything

            if (id > arenas.Count)
            {
                Debug.LogError($"instancenumber {id} is bigger than arenas.Count {arenas.Count}");
                // this is programmed to always receive incrementing instancenumbers
            }

            Vector3 arenaPosition = this.arenaPosition + id * arenaOffset;


            GameObject arenaGameObject = Instantiate(arenaPrefab, arenaPosition, Quaternion.identity);
            arenaGameObject.name = $"Arena {id}";

            Arena arena = arenaGameObject.GetComponent<Arena>();
            arena.setInstanceNumber(id);

            arenas.Add(arena);

        }

        [JsonRpcMethod]
        string getObservation(int id)
        {
            return arenas[id].getObservation();
        }

        [JsonRpcMethod]
        string getArenaScreenshot(int id)
        {
            return arenas[id].getArenaScreenshot();
        }

    }

    Rpc rpc;

    public GameObject arenaPrefab;

    public Vector3 arenaOffset = new Vector3(0, 0, 30);

    void Awake()
    {
        rpc = new Rpc(arenaPrefab, arenaOffset);
        print("rpc started");
    }

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }
}