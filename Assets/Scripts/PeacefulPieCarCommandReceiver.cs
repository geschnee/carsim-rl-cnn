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
        AIEngine aIEngine;
        Camera carCam;

        bool simRunning = true;

        public Rpc(PeacefulPieCarCommandReceiver sphere, GameObject car, Camera carCam)
        {
            this.sphere = sphere;
            //this.car = car;
            this.aIEngine = car.GetComponent<AIEngine>();
            this.carCam = carCam;
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
            Debug.Log($"forwrd left {inputAccelerationLeft} right {inputAccelerationRight}");
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
            Debug.Log("getObservation");
            string cameraPicture = GetCameraInput();
            return cameraPicture;
        }

        int resWidth = 240;
        int resHeight = 240;

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


            carCam.targetTexture = null;
            RenderTexture.active = null; // JC: added to avoid errors
            Destroy(rt);
            Destroy(screenShot);

            File.WriteAllBytes("observation.png", pictureInBytes);
            File.WriteAllBytes("observation.txt", pictureInBytes);

            Debug.Log($"Shape of byte[] {pictureInBytes.Length}");
            Debug.Log($"byte[] {pictureInBytes}");
            Debug.Log($"first byte: {pictureInBytes[0]}");

            // test as strings, since the byte array received has a different length in python than in c# anyways
            string utfString = Encoding.ASCII.GetString(pictureInBytes, 0, pictureInBytes.Length);
            Debug.Log($"uft-8 string {utfString}");
            File.WriteAllText("encoded_string.txt", utfString);
            // utfString has the same length as the one received in python!!!
            // and same content, now how to get the string back to png?

            // System.Byte[]   pictureInBytes;
            return utfString;
        }
    }

    Rpc rpc;
    public GameObject car;
    public Camera carCam;

    void Awake()
    {
        rpc = new Rpc(this, car, carCam);
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