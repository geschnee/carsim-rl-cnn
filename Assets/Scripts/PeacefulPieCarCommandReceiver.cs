using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AustinHarris.JsonRpc;

using System.Text;

using System.IO;
using System.Linq;
public class PeacefulPieCarCommandReceiver : MonoBehaviour
{
    class Rpc : JsonRpcService
    {
        PeacefulPieCarCommandReceiver sphere;
        //GameObject car;

        GameManager gameManager;

        GameObject carPrefab;
        GameObject car;
        Camera carCam;

        AIEngine aIEngine;

        bool simRunning = true;

        public Rpc(PeacefulPieCarCommandReceiver sphere, GameObject prefab, GameObject gameManagerObject)
        {
            this.gameManager = gameManagerObject.GetComponent<GameManager>();
            this.sphere = sphere;

            this.carPrefab = prefab;
        }

        [JsonRpcMethod]
        void spawnCar()
        {
            Debug.Log($"spawnCar");

            if (this.car != null)
            {
                Debug.Log($"will destroy existing car");
                Destroy(this.car);
            }

            //GameObject car = Instantiate(carPrefab, new Vector3(10, 1, 60), Quaternion.identity);

            GameObject car = gameManager.spawnJetbot();

            this.car = car;

            this.aIEngine = car.GetComponent<AIEngine>();
            this.carCam = car.GetComponentInChildren<Camera>();
        }

        [JsonRpcMethod]
        void startEpisode()
        {
            Debug.Log($"startEpisode");
            car.GetComponent<EpisodeManager>().StartEpisode();
        }

        [JsonRpcMethod]
        void say(string message)
        {
            Debug.Log($"you sent {message}");
        }

        [JsonRpcMethod]
        float getHeight()
        {
            Debug.Log($"get height triggered");
            return sphere.transform.position.y;
        }

        [JsonRpcMethod]
        MyVector3 getPosition()
        {
            return new MyVector3(sphere.transform.position);
        }

        [JsonRpcMethod]
        void translate(MyVector3 translate)
        {
            sphere.transform.position += translate.ToVector3();
        }

        [JsonRpcMethod]
        void forwardInputsToCar(float inputAccelerationLeft, float inputAccelerationRight)
        {
            //Debug.Log($"forward left {inputAccelerationLeft} right {inputAccelerationRight}");
            aIEngine.SetInput(inputAccelerationLeft, inputAccelerationRight);

        }

        [JsonRpcMethod]
        void pauseStartSimulation()
        {
            Debug.Log($"pauseStart simRunning {simRunning}");
            if (simRunning == true)
            {
                simRunning = false;
                Time.timeScale = 2f; // test fo thr lolz
                // when pausing completely (timeScale = 0) the entire simulation stops and cannot go back to full tim, this is here not triggered
                // https://github.com/Astn/JSON-RPC.NET/blob/master/Json-Rpc/JsonRpcService.cs
            }
            else
            {
                simRunning = true;
                Time.timeScale = 1;
            }
        }

        [JsonRpcMethod]
        string getObservation()
        {
            if (car == null)
            {
                Debug.Log($"car is null in getObservation, try spawn new one");
                spawnCar();
            }

            //Debug.Log("getObservation");
            string cameraPicture = GetCameraInput();
            return cameraPicture;
        }

        int resWidth = 512; // from CarAgent.cs
        int resHeight = 256;
        // resolution is quite high: https://www.raspberrypi.com/documentation/accessories/camera.html

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
    }

    Rpc rpc;
    public GameObject carPrefab;
    public GameObject gameManager;


    void Awake()
    {
        rpc = new Rpc(this, carPrefab, gameManager);
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