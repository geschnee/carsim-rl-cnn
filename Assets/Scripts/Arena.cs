using UnityEngine;
using System.Collections;
using System.Collections.Generic;

using System.Text;

using System.IO;
using System.Linq;

public class Arena : MonoBehaviour
{
    // TODO in theory this class could replace the GameManager class in it's responsibilities

    EpisodeManager episodeManager;

    public GameObject gameManagerObject;

    GameManager gameManager;

    GameObject car;
    Camera carCam;

    AIEngineBase aIEngine;

    private int instancenumber;

    // from CarAgent.cs width was 512 and height was 256
    // we reduce the size to make it easier for the python code to handle the images
    // so more fit in the replay buffer
    // these parameters are set in python config now, we want to be able to control the resolution and width/height ratio
    public int resWidth; // = 336;
    public int resHeight; // = 168;
    // resolution is quite high: https://www.raspberrypi.com/documentation/accessories/camera.html

    public Camera arenaCam;
    public int arenaResWidth = 512;
    public int arenaResHeight = 512;

    public List<Light> lights;

    public float velocityCoefficient;
    public float orientationCoefficient;
    public float distanceCoefficient;
    public float eventCoefficient;

    public bool fixedTimesteps;
    public float fixedTimestepsLength;

    public VideoRecorder arenaRecorder;

    public Material skyboxMaterialBright;
    public Material skyboxMaterialDark;
    public Material skyboxMaterialDefault;

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
            Destroy(this.car);
        }

        // destroy previous obstacles:
        gameManager.DestroyObstaclesOnMap();
    }

    public string reset(MapType mt, Spawn jetBotSpawn, LightSetting lightSetting, string video_filename)
    {
        if (this.car != null)
        {
            this.car.SetActive(false);
            // else there was strange behaviour when the new objects were spawned
            // it looked like there was collision detection for the Destroyed car
            Destroy(this.car);
        }

        // destroy previous obstacles:
        gameManager.DestroyObstaclesOnMap();

        // spawn new obstacles:
        MapData md = gameManager.InitializeMapWithObstacles(mt, 0, jetBotSpawn);

        GameObject car = gameManager.spawnJetbot(md, this.instancenumber);



        this.car = car;


        this.aIEngine = car.GetComponent<AIEngineBase>();
        this.aIEngine.ResetMotor();

        this.carCam = car.GetComponentInChildren<Camera>();
        

        SetLightSetting(lightSetting);

        episodeManager = car.GetComponent<EpisodeManager>();

        episodeManager.velocityCoefficient = this.velocityCoefficient;
        episodeManager.orientationCoefficient = this.orientationCoefficient;
        episodeManager.distanceCoefficient = this.distanceCoefficient;
        episodeManager.eventCoefficient = this.eventCoefficient;

        episodeManager.fixedTimesteps = fixedTimesteps;
        episodeManager.fixedTimestepsLength = fixedTimestepsLength;

        episodeManager.StartEpisode();
        episodeManager.arenaRecorder = arenaRecorder;

        VideoRecorder jetBotRecorder = car.GetComponent<VideoRecorder>();
        episodeManager.jetBotRecorder = jetBotRecorder;

        if (video_filename != "")
        {
            //Debug.Log($"start video recording");
            arenaRecorder.episodeManager = episodeManager;
            arenaRecorder.StartVideo(video_filename);

            jetBotRecorder.episodeManager = episodeManager;
            jetBotRecorder.StartVideo(video_filename + "_jetbot");
        }


        return this.getObservation();
    }

    public void SetLightSetting(LightSetting lightSetting)
    {
        float lightMultiplier;
        if (lightSetting == LightSetting.bright)
        {
            lightMultiplier = 7.5f;
        } else if (lightSetting == LightSetting.standard)
        {
            lightMultiplier = 5f;
        } else if (lightSetting == LightSetting.dark)
        {
            lightMultiplier = 2.5f;
        } else {

            lightMultiplier = -100;
            Debug.LogError("LightSetting random should not be used");
        }

        foreach (Light light in lights)
        {
            light.intensity = lightMultiplier;
        }

        // This sets the skybox material of the agent's camera based on the lightMultiplier
        Skybox skybox = carCam.GetComponent<Skybox>();

        if (lightMultiplier < 4)
        {
            skybox.material = skyboxMaterialDark;
        }
        else if (lightMultiplier > 6)
        {
            skybox.material = skyboxMaterialBright;
        }
        else
        {
            skybox.material = skyboxMaterialDefault;
        }
    }

    public void forwardInputsToCar(float inputAccelerationLeft, float inputAccelerationRight)
    {
        //Debug.Log($"forward left {inputAccelerationLeft} right {inputAccelerationRight}");

        aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);

    }

    public EpisodeManager getEpisodeManager()
    {
        return episodeManager;
    }

    public StepReturnObject immediateStep(int step, float inputAccelerationLeft, float inputAccelerationRight)
    {
        // TODO maybe move this code to the episodeManager

        if (episodeManager.fixedTimesteps && episodeManager.IsTerminated()==false){
            if (episodeManager.episodeStatus != EpisodeStatus.WaitingForStep){
                return new StepReturnObject(previousStepNotFinished: true);
            }
        }


        // Debug.LogWarning($"immediateStep {step} {inputAccelerationLeft} {inputAccelerationRight} for {this.car.name}");
        // when the error happens is the other input the same?
        aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);
        episodeManager.IncreaseSteps(step);


        bool done = episodeManager.IsTerminated();
        bool terminated = episodeManager.IsTerminated();
        string observation = this.getObservation();


        Dictionary<string, string> info = episodeManager.GetInfo();
        List<float> rewards = episodeManager.GetRewards();

        return new StepReturnObject(observation, done, terminated, info, rewards);
    }

    public void setJetbot(string jetbotName)
    {
        //Debug.Log($"setJetbot {jetbotName}");
        gameManager.setJetbot(jetbotName);
    }

  
  


    //Get the AI vehicles camera input encode as byte array
    private string GetCameraInput(Camera cam, int resWidth, int resHeight, string filename)
    {
        // TODO should the downsampling to 84 x 84 happen here instead of python?
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        cam.targetTexture = rt;
        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        cam.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);

        System.Byte[] pictureInBytes = screenShot.EncodeToPNG(); // the length of the array changes depending on the content
                                                                 // screenShot.EncodeToJPG(); auch möglich

        cam.targetTexture = null;
        RenderTexture.active = null; // JC: added to avoid errors
        Destroy(rt);
        Destroy(screenShot);

        File.WriteAllBytes(filename, pictureInBytes);

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
        return GetCameraInput(this.carCam, this.resWidth, this.resHeight, "observation.png");
    }

    public string getArenaScreenshot()
    {
        string cameraPicture = GetCameraInput(this.arenaCam, this.arenaResWidth, this.arenaResHeight, "arena.png");
        return cameraPicture;
    }

}