using UnityEngine;
using System.Collections;
using System.Collections.Generic;

using System.Text;

using System.IO;
using System.Linq;

public class Arena : MonoBehaviour
{
    // TODO in theory this class could replace the GameManager class in it's responsibilities

    public GameObject gameManagerObject;

    GameManager gameManager;

    GameObject car;
    Camera carCam;

    AIEngine aIEngine;

    private float rewardAsync = 0f;

    private int instancenumber;

    int resWidth = 512; // from CarAgent.cs
    int resHeight = 256;
    // resolution is quite high: https://www.raspberrypi.com/documentation/accessories/camera.html

    void Awake()
    {
        // initialize new arena at the correct position

        // initialize the private variables

        this.gameManager = gameManagerObject.GetComponent<GameManager>();

    }

    public void setInstanceNumber(int instancenumber)
    {
        this.instancenumber = instancenumber;
    }

    public void destroyMap()
    {
        if (this.car != null)
        {
            Debug.Log($"will destroy existing car");
            Destroy(this.car);
        }

        // destroy previous obstacles:
        gameManager.DestroyObstaclesOnMap();
    }

    public string reset(MapType mt, bool jetBotSpawnpointRandom, bool singleGoalTraining)
    {
        if (this.car != null)
        {
            Debug.Log($"will destroy existing car");
            this.car.SetActive(false);
            // else there was strange behaviour when the new objects were spawned
            // it looked like there was collision detection for the Destroyed car
            Destroy(this.car);
        }

        // destroy previous obstacles:
        gameManager.DestroyObstaclesOnMap();

        Debug.Log($"startEpisode");

        // spawn new obstacles:
        MapData md = gameManager.InitializeMapWithObstacles(mt, 0, jetBotSpawnpointRandom, singleGoalTraining);


        GameObject car = gameManager.spawnJetbot(md);

        this.car = car;

        this.aIEngine = car.GetComponent<AIEngine>();
        this.carCam = car.GetComponentInChildren<Camera>();



        car.GetComponent<EpisodeManager>().StartEpisode();

        // TODO does reset need to return something?
        // TOOD yes, need to return an observation

        return GetCameraInput();
    }

    public void forwardInputsToCar(float inputAccelerationLeft, float inputAccelerationRight)
    {
        //Debug.Log($"forward left {inputAccelerationLeft} right {inputAccelerationRight}");
        aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);

    }

    public StepReturnObject immediateStep(float inputAccelerationLeft, float inputAccelerationRight)
    {
        // TODO maybe move this code to the episodeManager
        aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);

        float reward = car.GetComponent<EpisodeManager>().GetReward();
        bool done = car.GetComponent<EpisodeManager>().IsTerminated();
        bool terminated = car.GetComponent<EpisodeManager>().IsTerminated();
        string observation = GetCameraInput();

        Dictionary<string, string> info = car.GetComponent<EpisodeManager>().GetInfo();

        return new StepReturnObject(observation, reward, done, terminated, info);
    }

    public void asyncStepPart1(float inputAccelerationLeft, float inputAccelerationRight)
    {
        aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);
        rewardAsync = car.GetComponent<EpisodeManager>().rewardSinceLastGetReward;
        // part1 sets the actions, python does the waiting, then part2 returns the observation
    }

    public StepReturnObject asyncStepPart2()
    {
        float reward = car.GetComponent<EpisodeManager>().GetReward();
        //Debug.Log($"reward diff: {reward - rewardAsync}");


        bool done = car.GetComponent<EpisodeManager>().IsTerminated();
        bool terminated = car.GetComponent<EpisodeManager>().IsTerminated();
        string observation = GetCameraInput();

        Dictionary<string, string> info = car.GetComponent<EpisodeManager>().GetInfo();

        return new StepReturnObject(observation, reward, done, terminated, info);
    }




    //Get the AI vehicles camera input encode as byte array
    private string GetCameraInput()
    {
        RenderTexture rt = new RenderTexture(this.resWidth, this.resHeight, 24);
        carCam.targetTexture = rt;
        Texture2D screenShot = new Texture2D(this.resWidth, this.resHeight, TextureFormat.RGB24, false);
        carCam.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);

        System.Byte[] pictureInBytes = screenShot.EncodeToPNG(); // the length of the array changes depending on the content
                                                                 // screenShot.EncodeToJPG(); auch möglich

        carCam.targetTexture = null;
        RenderTexture.active = null; // JC: added to avoid errors
        Destroy(rt);
        Destroy(screenShot);

        File.WriteAllBytes("observation.png", pictureInBytes);



        string base64_string = System.Convert.ToBase64String(pictureInBytes);


        byte[] base64EncodedBytes = System.Convert.FromBase64String(base64_string);

        /*
        Debug.Log($"base64_string {base64_string}");

        Debug.Log($"Shape of byte[] {pictureInBytes.Length}");
        Debug.Log($"byte[] {pictureInBytes}");
        Debug.Log($"first byte: {pictureInBytes[0]}");

        Debug.Log($"base64EncodedBytes {base64EncodedBytes}");
        Debug.Log($"base64EncodedBytes length {base64EncodedBytes.Length}");
        Debug.Log($"base64EncodedBytes first char {base64EncodedBytes[0]}");
        */

        return base64_string;
    }

    public string getObservation()
    {
        if (car == null)
        {
            // car is not spawned yet, give some default image
            return DefaultImage.getDefaultImage();
        }

        //Debug.Log("getObservation");
        string cameraPicture = GetCameraInput();
        return cameraPicture;
    }
}