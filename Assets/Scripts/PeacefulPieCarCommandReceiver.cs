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
        this.rewards = new List<float>();
        this.done = false;
        this.terminated = false;
        this.info = new Dictionary<string, string>();
        this.observation = "stepWaiting";
    }
}

public class StepReturnObjectList
{
    public List<StepReturnObject> objects = new List<StepReturnObject>();
    public float step_script_realtime_duration;
}


public class PeacefulPieCarCommandReceiver : MonoBehaviour
{
    class Rpc : JsonRpcService
    {

        GameObject arenaPrefab;

        private List<Arena> arenas = new List<Arena>();

        private Vector3 arenaPosition = new Vector3(0, 0, 0);
        private Vector3 arenaOffset;

        private float step_script_realtime_duration;

        public Rpc(GameObject arenaPrefab, Vector3 arenaOffset)
        {
            this.arenaPrefab = arenaPrefab;
            this.arenaOffset = arenaOffset;
            step_script_realtime_duration = 0.0f;
        }

        [JsonRpcMethod]
        void destroyMap(int id)
        {
            arenas[id].destroyMap();
            step_script_realtime_duration = 0.0f;
        }

        [JsonRpcMethod]
        void deleteAllArenas()
        {
            foreach (Arena arena in arenas)
            {
                Destroy(arena.gameObject);
            }
            arenas.Clear();
            step_script_realtime_duration = 0.0f;
        }


        [JsonRpcMethod]
        string reset(int id, string mapType, string spawn, string lightSettingName, string video_filename)
        {
            //Debug.Log($"reset() called, id: {id}");
            MapType mt = (MapType)Enum.Parse(typeof(MapType), mapType);
            Spawn sp = (Spawn)Enum.Parse(typeof(Spawn), spawn);
            LightSetting lightSetting = (LightSetting)Enum.Parse(typeof(LightSetting), lightSettingName);

            //Debug.Log($"mt: {mt}");
            return arenas[id].reset(mt, sp, lightSetting, video_filename);
        }

        [JsonRpcMethod]
        void say(string message)
        {
            Debug.Log($"you sent {message}");
        }

        [JsonRpcMethod]
        int ping(int id)
        {
            //Debug.Log($"you sent {message}");
            return 0;
        }

        [JsonRpcMethod]
        void forwardInputsToCar(int id, float inputAccelerationLeft, float inputAccelerationRight)
        {
            arenas[id].forwardInputsToCar(inputAccelerationLeft, inputAccelerationRight);
        }

        [JsonRpcMethod]
        StepReturnObject immediateStep(int id, int step, float inputAccelerationLeft, float inputAccelerationRight)
        {
            float beforeTime = Time.realtimeSinceStartup;
            StepReturnObject obj = arenas[id].step(step, inputAccelerationLeft, inputAccelerationRight);
            float step_script_realtime = Time.realtimeSinceStartup - beforeTime;
            step_script_realtime_duration += step_script_realtime;
            return obj;
        }

        [JsonRpcMethod]
        StepReturnObjectList bundledStep(List<int> step_nrs, List<float> left_actions, List<float> right_actions)
        {
            float beforeTime = Time.realtimeSinceStartup;
            StepReturnObjectList objects = new StepReturnObjectList();

            for (int i = 0; i < left_actions.Count; i++)
            {
                StepReturnObject stepReturnObject = arenas[i].step(step_nrs[i], left_actions[i], right_actions[i]);
                objects.objects.Add(stepReturnObject);
            }
            float step_script_realtime = Time.realtimeSinceStartup - beforeTime;
            step_script_realtime_duration += step_script_realtime;

            objects.step_script_realtime_duration = step_script_realtime_duration;

            return objects;
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
        string[] getObservationAllEnvs()
        {
            List<string> observations = new List<string>();
            foreach (Arena arena in arenas)
            {
                observations.Add(arena.getObservation());
            }

            return observations.ToArray();
        }

        [JsonRpcMethod]
        System.Byte[] getObservationBytes(int id)
        {
            return arenas[id].getObservationBytes();
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