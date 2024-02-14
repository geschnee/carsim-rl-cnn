using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;
using System.Drawing;
// unity does not support the System.drawing library: https://docs.unity3d.com/Manual/overview-of-dot-net-in-unity.html#:~:text=Unity%20uses%20the%20open%2Dsource,variety%20of%20different%20hardware%20configurations.


using ImageMagick;
// steps to making ImageMagick work in Unity:
// use NuGet to install Magick.NET-Q16-AnyCPU
// use NuGet to install Magick.NET.Core
// download Magick.Native-Q16-x64.dll from internet and place it in the unity packages directory
// example download link: https://www.dllme.com/dll/files/magick_native-q16-x64/e924369b24e1de791993f37f0aad1e3c

public class VideoRecorder : MonoBehaviour
{
    
    float fps = 6;
    public EpisodeManager episodeManager;

    public List<byte[]> frames; 

    public Camera camera; // arena Camera
    
    public string video_filename;

    public int fileCounter = 0;

    float lastRecordTime;

    public bool isRecording;

    public void StartVideo(string video_filename)
    {
        this.video_filename = video_filename;
        this.frames = new List<byte[]>();

        this.fileCounter=0;

        lastRecordTime = 0; // this will result in an immediate Capture in Update

        // https://forum.unity.com/threads/how-to-save-manually-save-a-png-of-a-camera-view.506269/

        this.isRecording = true;
    }

    public void Capture()
    {
        
        //Debug.Log("Captured frame " + fileCounter + " at " + Time.time + " seconds by " + this.transform.name);
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = camera.targetTexture;
 
        camera.Render();
 
        Texture2D image = new Texture2D(camera.targetTexture.width, camera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, camera.targetTexture.width, camera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;
 
        //byte[] bytes = image.EncodeToPNG();
        frames.Add(image.EncodeToPNG());
 
        //File.WriteAllBytes(Application.dataPath + "/../Images/" + video_filename + "_" + fileCounter + ".png", bytes);
        //fileCounter++;



        lastRecordTime = Time.time;
        
        Destroy(image);
    }

    public void StopVideo()
    {
        if (!this.isRecording)
        {
            Debug.LogError("Video already stopped");
            return;
        }


        this.isRecording = false;
        Debug.Log("Video stopped");
        // TODO implement video saving

        // use gif instead, much easier
        // https://stackoverflow.com/questions/1196322/how-to-create-an-animated-gif-in-net

        saveGif();
        //destroyImages();
    }

    private void destroyImages()
    {
        // we need to destoy the images ourselves: https://forum.unity.com/threads/texture2d-destroy-how-to-do-it-right.385249/
        //foreach (Texture2D frame in frames)
        //{
        //    Destroy(frame);
        //}
    }

    private void saveGif(){

        //MagickImage image = new MagickImage("Snakeware.png");
        float f_delay = 1 / fps * 1000;
        int delay = (int) f_delay; // in milliseconds

        using (MagickImageCollection collection = new MagickImageCollection())
        {
            // Add first image and set the animation delay to 100ms

            for (int i = 0; i < frames.Count; i++)
            {
                MagickImage img = new MagickImage(frames[i]);
                collection.Add(img);
                collection[i].AnimationDelay = delay;
            }
            //MagickImage img = new MagickImage(frames[0].EncodeToPNG());
            //collection.Add(img);
            //collection[0].AnimationDelay = delay;

            // Add second image, set the animation delay to 100ms and flip the image
            //MagickImage img2 = new MagickImage(frames[0].EncodeToPNG());
            //collection.Add(img2);
            //collection[1].AnimationDelay = delay;
            //collection[1].Flip();

            // Optionally reduce colors
            QuantizeSettings settings = new QuantizeSettings();
            settings.Colors = 256;
            collection.Quantize(settings);

            // Optionally optimize the images (images should have the same size).
            collection.Optimize();

            // Save gif
            collection.Write($"{video_filename}.gif");
        }

    }

    public void FixedUpdate()
    {
        if (!isRecording)
        {
            return;
        }

        if (this.episodeManager.episodeRunning == false)
        {
            return;
        }
        if (Time.time - lastRecordTime > 1.0f / fps)
        {
            Capture();
        }
    }

}
