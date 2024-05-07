using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class AIEngineBase : MonoBehaviour
{
    public float inputAccelerationLeft = 0;
    public float inputAccelerationRight = 0;

    public Transform carBody;

    EpisodeManager episodeManager;

    public void Start()
    {
        this.episodeManager = this.GetComponent<EpisodeManager>();
    }

    public void FixedUpdate()
    {
        if (!episodeManager.isEpisodeRunning())
        {
            ResetMotor();
            return;
        }
        this.HandleMotor();
        this.UpdateWheels();
    }

    public void SetInput(float inputAccelerationLeft, float inputAccelerationRight)
    {
        // normal input
        this.inputAccelerationLeft = inputAccelerationLeft;
        this.inputAccelerationRight = inputAccelerationRight;
    }

    public abstract void HandleMotor();

    public abstract void ResetMotor();

    public virtual void UpdateWheels()
    {
        Debug.LogError($"UpdateWheels of AIEngineBase called");
    }

    public virtual float getCarVelocity()
    {
        Debug.LogError($"getCarVelocity of AIEngineBase called");
        return 0;
    }
}
