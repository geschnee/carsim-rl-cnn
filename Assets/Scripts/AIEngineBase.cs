using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public abstract class AIEngineBase : MonoBehaviour
{
    public float inputAccelerationLeft = 0;
    public float inputAccelerationRight = 0;

    public bool episodeRunning = false;

    public Transform carBody;

    public void FixedUpdate()
    {
        if (!episodeRunning)
        {
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
        episodeRunning = true;
    }

    public virtual void HandleMotor()
    {
        Debug.LogError($"HandleMotor of AIEngineBase called");
    }

    public virtual void ResetMotor()
    {
        Debug.LogError($"ResetMotor of AIEngineBase called");
    }

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
