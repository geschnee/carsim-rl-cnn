using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;



using System.Threading;

using ImageMagick;
// steps to making ImageMagick work in Unity:
// it worked with version 13.5.0
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
    public float episodeDuration;
    public VideoData(string video_filename, List<float> delays, List<byte[]> frames, float episodeDuration)
    {
        this.video_filename = video_filename;
        this.delays = delays;
        this.frames = frames;
        this.episodeDuration = episodeDuration;
    }
}

public class VideoRecorder : MonoBehaviour
{

    float target_fps = 20;
    public EpisodeManager episodeManager;

    public VideoData videoData;

    public Camera cam;

    float lastRecordTime;

    public bool isRecording;

    public RenderTexture renderTexture;


    public void StartVideo(string video_filename_in)
    {


        lastRecordTime = -1; // this will result in an immediate Capture in Update



        videoData = new VideoData(video_filename_in, new List<float>(), new List<byte[]>(), 0);


        this.cam.targetTexture = renderTexture;


        this.isRecording = true;

    }

    public void Capture()
    {
        this.cam.targetTexture = renderTexture;
        // for some reason the target texture of the agent camera always resettet itself to null
        // the arena cam always worked fine


        // https://forum.unity.com/threads/how-to-save-manually-save-a-png-of-a-camera-view.506269/
        RenderTexture activeRenderTexture = RenderTexture.active;
        RenderTexture.active = cam.targetTexture;

        cam.Render();

        Texture2D image = new Texture2D(cam.targetTexture.width, cam.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, cam.targetTexture.width, cam.targetTexture.height), 0, 0);
        image.Apply();
        RenderTexture.active = activeRenderTexture;

        this.videoData.frames.Add(image.EncodeToPNG());

        if (lastRecordTime != -1)
        {
            this.videoData.delays.Add(this.episodeManager.duration - lastRecordTime);
        }
        lastRecordTime = this.episodeManager.duration;

        Destroy(image);
    }

    void OnApplicationQuit()
    {
        if (this.isRecording)
        {
            Debug.Log("Application quit while recording video. Stopping video recording.");
            StopVideo(this.episodeManager.duration);
        }
    }

    public void StopVideo(float episodeDuration)
    {
        if (!this.isRecording)
        {
            // Video already stopped or was never started (EpisodeManager calls this method regardless of Video started or not)
            return;
        }

        this.videoData.episodeDuration = episodeDuration;

        this.isRecording = false;


        // use thread to prevent blocking (optimize gif step takes a long time)
        Thread t = new Thread(() => saveGif(this.videoData));
        t.Start();
    }

    void saveGif(VideoData videoData)
    {
        // use gif instead of mp4, much easier/inexpensive
        // https://stackoverflow.com/questions/1196322/how-to-create-an-animated-gif-in-net


        float f_delay_in_milliseconds = 1 / videoData.fps * 1000;
        int delay = (int)f_delay_in_milliseconds;

        float gif_duration_in_seconds = 0;

        if (videoData.frames.Count != videoData.delays.Count + 1)
        {
            if (videoData.frames.Count == 0 && videoData.delays.Count == 0)
            {
                return;
            }
            else
            {
                Debug.LogError($"Frames and delays do not match in length. Frames: {videoData.frames.Count}, Delays: {videoData.delays.Count} for {this.gameObject.name}");
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
                    collection[i].AnimationDelay = (int)videoData.delays[i] * 1000;
                    gif_duration_in_seconds += videoData.delays[i];
                }
                else
                {
                    // add a "default delay"
                    collection[i].AnimationDelay = delay;
                }
            }

            // https://github.com/search?q=repo%3Adlemstra%2FMagick.NET%20loop&type=code
            collection[0].AnimationIterations = 1; // repeat only once



            // Optionally reduce colors
            QuantizeSettings settings = new QuantizeSettings();
            settings.Colors = 256;

            collection.Quantize(settings);

            collection.Optimize();

            collection.Write($"{videoData.video_filename}.gif");
        }
        // Debug.Log($"Gif length: {videoData.frames.Count} frames {gif_duration_in_seconds} seconds. Episode length {videoData.episodeDuration} seconds. Saved to {videoData.video_filename}.gif");
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
