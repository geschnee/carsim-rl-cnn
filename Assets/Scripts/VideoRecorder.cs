using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;
using System.Drawing;
// unity does not support the System.drawing library: https://docs.unity3d.com/Manual/overview-of-dot-net-in-unity.html#:~:text=Unity%20uses%20the%20open%2Dsource,variety%20of%20different%20hardware%20configurations.


using System.Threading;

using ImageMagick;
// steps to making ImageMagick work in Unity:
// use NuGet to install Magick.NET-Q16-AnyCPU
// use NuGet to install Magick.NET.Core
// download Magick.Native-Q16-x64.dll from internet and place it in the unity packages directory
// example download link: https://www.dllme.com/dll/files/magick_native-q16-x64/e924369b24e1de791993f37f0aad1e3c

public class VideoData
{
    public string video_filename;
    public float fps;
    public List<float> delays;
    public List<byte[]> frames;
    public float gameDuration;
    public VideoData(string video_filename, List<float> delays, List<byte[]> frames, float gameDuration)
    {
        this.video_filename = video_filename;
        this.delays = delays;
        this.frames = frames;
        this.gameDuration = gameDuration;
    }
}

public class VideoRecorder : MonoBehaviour
{
    
    float target_fps = 20;
    public EpisodeManager episodeManager;

    public VideoData videoData;

    public Camera arenaCamera;
    
    public string video_filename="";

    public int fileCounter = 0;

    float lastRecordTime;

    public bool isRecording;


    public void StartVideo(string video_filename_in)
    {

        if (video_filename_in != this.video_filename) {
            this.fileCounter = 0;
            this.video_filename = video_filename_in;
            // filename changed --> reset fileCounter
        }

        lastRecordTime = -1; // this will result in an immediate Capture in Update

        this.isRecording = true;

        videoData = new VideoData(video_filename_in + this.fileCounter, new List<float>(), new List<byte[]>(), 0);
        this.fileCounter++;

    }

    public void Capture()
    {
        if (lastRecordTime != -1)
        {
            this.videoData.delays.Add(this.episodeManager.duration - lastRecordTime);
        }
        lastRecordTime = this.episodeManager.duration;


        // https://forum.unity.com/threads/how-to-save-manually-save-a-png-of-a-camera-view.506269/
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = arenaCamera.targetTexture;
 
        arenaCamera.Render();
 
        Texture2D image = new Texture2D(arenaCamera.targetTexture.width, arenaCamera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, arenaCamera.targetTexture.width, arenaCamera.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;
 
        this.videoData.frames.Add(image.EncodeToPNG());
        
        
        Destroy(image);
    }

    public void StopVideo(float gameDuration)
    {
        if (!this.isRecording)
        {
            // Video already stopped or was never started (EpisodeManager calls this method regardless of Video started or not)
            return;
        }

        this.videoData.gameDuration = gameDuration;

        this.isRecording = false;

        
        // use thread to prevent blocking (optimize gif step takes a long time)
        Thread t = new Thread( ()=>saveGif(this.videoData));
        t.Start();
    }

    void saveGif(VideoData videoData){
        // use gif instead of mp4, much easier/inexpensive
        // https://stackoverflow.com/questions/1196322/how-to-create-an-animated-gif-in-net


        float f_delay_in_milliseconds = 1 / videoData.fps * 1000;
        int delay = (int) f_delay_in_milliseconds;

        float writeTime=0;
        float addTime=0;
        float optimizeTime=0;

        float gif_duration_in_seconds = 0;

        if (videoData.frames.Count != videoData.delays.Count + 1)
        {
            if (videoData.frames.Count == 0 && videoData.delays.Count == 0)
            {
                return;
            }
            else
            {
                Debug.LogError($"Frames and delays do not match in length. Frames: {videoData.frames.Count}, Delays: {videoData.delays.Count}");
            }
            
        }

        using (MagickImageCollection collection = new MagickImageCollection())
        {

            for (int i = 0; i < videoData.frames.Count; i++)
            {
                MagickImage img = new MagickImage(videoData.frames[i]);
                collection.Add(img);

                if (i < videoData.frames.Count - 1)
                {
                    // add the recorded delay to the gif only if it is not the last frame (since we do not have a recorded delay there)
                    collection[i].AnimationDelay = (int) videoData.delays[i] * 1000;
                    gif_duration_in_seconds += videoData.delays[i];
                } else {
                    // add a "default delay"
                    collection[i].AnimationDelay = delay;
                }
            }
        
            // Optionally reduce colors
            QuantizeSettings settings = new QuantizeSettings();
            settings.Colors = 256;

            collection.Quantize(settings);

            collection.Optimize();
           
            collection.Write($"{videoData.video_filename}.gif");
        }
       
        
        Debug.Log($"Gif length: {videoData.frames.Count} frames {gif_duration_in_seconds} seconds. Game length {videoData.gameDuration} seconds. Saved to {videoData.video_filename}.gif");
    }

    public void FixedUpdate()
    {
        if (!isRecording)
        {
            return;
        }

        if (this.episodeManager.episodeStatus != EpisodeStatus.Running)
        {
            return;
        }
        if (this.episodeManager.duration - this.lastRecordTime > 1.0f / target_fps)
        {
            Capture();
        }
    }

}
