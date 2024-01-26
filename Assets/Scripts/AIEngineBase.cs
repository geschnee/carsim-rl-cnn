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

    public void Start()
    {
    }

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
        // + - 10 %
        //this.inputAccelerationLeft = (float)(input[1]);
        //this.inputAccelerationRight = (float) (input[0]);

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

    }


    public virtual float getSteeringAngle()
    {
        Debug.LogError($"getSteeringAngle of AIEngineBase called");

        return this.carBody.eulerAngles.y;
    }

    public virtual float getCarVelocity()
    {

        Debug.LogError($"getCarVelocity of AIEngineBase called");

        return 0;
    }
}
