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
    public bool previousStepNotFinished;

    public string observation;
    public bool done;
    public bool terminated;

    public Dictionary<string, string> info;

    public List<float> rewards;

    public StepReturnObject(string observation, bool done, bool terminated, Dictionary<string, string> info, List<float> rewards)
    {
        this.observation = observation;
        this.done = done;
        this.terminated = terminated;
        this.info = info;
        this.rewards = rewards;
        this.previousStepNotFinished = false;
    }

    public StepReturnObject(bool previousStepNotFinished)
    {
        this.previousStepNotFinished = previousStepNotFinished;
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
        string reset(int id, string mapType, string spawn, float lightMultiplier, string video_filename)
        {
            //Debug.Log($"reset() called, id: {id}");
            MapType mt = (MapType)Enum.Parse(typeof(MapType), mapType);
            Spawn sp = (Spawn)Enum.Parse(typeof(Spawn), spawn);
            //Debug.Log($"mt: {mt}");
            return arenas[id].reset(mt, sp, lightMultiplier, video_filename);
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
        StepReturnObject immediateStep(int id, int step, float inputAccelerationLeft, float inputAccelerationRight)
        {
            return arenas[id].immediateStep(step, inputAccelerationLeft, inputAccelerationRight);
        }

        [JsonRpcMethod]
        void startArena(int id, string jetbotName, float distanceCoefficient, float orientationCoefficient, float velocityCoefficient, float eventCoefficient, int resWidth, int resHeight, bool fixedTimesteps, float fixedTimestepsLength)
        {
            // Debug.Log($"startArena() called, id: {id}");
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
            arena.setJetbot(jetbotName);
            arena.fixedTimesteps = fixedTimesteps;
            arena.fixedTimestepsLength = fixedTimestepsLength;

            arenas.Add(arena);

            arena.distanceCoefficient = distanceCoefficient;
            arena.orientationCoefficient = orientationCoefficient;
            arena.velocityCoefficient = velocityCoefficient;
            arena.eventCoefficient = eventCoefficient;
            arena.resWidth = resWidth;
            arena.resHeight = resHeight;
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
        print($"{SystemInfo.processorCount} processors available");
        print($"{Environment.ProcessorCount} processors available");
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