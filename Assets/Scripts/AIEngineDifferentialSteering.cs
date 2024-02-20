using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIEngineDifferentialsteering : AIEngineBase
{
    private float maxTorque = 300f; // 100f;

    public WheelCollider frontLeftWheelCollider;
    public WheelCollider frontRightWheelCollider;

    public Transform frontLeftWheelTransform;
    public Transform frontRightWheeTransform;

    public override void HandleMotor()
    {
        float leftTorque = (inputAccelerationLeft * this.maxTorque);

        frontLeftWheelCollider.motorTorque = leftTorque;


        // Apply torque to the right wheel
        float rightTorque = (inputAccelerationRight * this.maxTorque);
        frontRightWheelCollider.motorTorque = rightTorque;

        // Differential steering does not change the steering angle of wheels        
    }

    public override void ResetMotor()
    {

        this.inputAccelerationLeft = 0;
        this.inputAccelerationRight = 0;
        frontLeftWheelCollider.steerAngle = 0;
        frontRightWheelCollider.steerAngle = 0;

        frontLeftWheelCollider.motorTorque = 0;
        frontRightWheelCollider.motorTorque = 0;

        frontLeftWheelCollider.attachedRigidbody.velocity = Vector3.zero;
        frontLeftWheelCollider.attachedRigidbody.angularVelocity = Vector3.zero;

        frontRightWheelCollider.attachedRigidbody.velocity = Vector3.zero;
        frontRightWheelCollider.attachedRigidbody.angularVelocity = Vector3.zero;

        HandleMotor();
        UpdateWheels();
    }

    public override void UpdateWheels()
    {
        UpdateSingleWheel(frontLeftWheelCollider, frontLeftWheelTransform);
        UpdateSingleWheel(frontRightWheelCollider, frontRightWheeTransform);
    }

    public void UpdateSingleWheel(WheelCollider wheelCollider, Transform wheelTransform)
    {
        Vector3 pos;
        Quaternion rot;
        wheelCollider.GetWorldPose(out pos, out rot);


        wheelTransform.rotation = rot;
        wheelTransform.position = pos;
        // TODO why do the wheels (mesh) rotate when the JetBot is fresh?
        // there has not been any command issued yet

        // sieht aus wie dieses Tutorial:
        // https://www.youtube.com/watch?v=rdl66506RY4&list=PL1R2qsKCcUCIdGXBLkZV2Tq_sxa-ADASN

    }


    /*public override float getSteeringAngle()
    {
        Debug.LogError("getSteeringAngle was called for differential steering");
        return this.carBody.eulerAngles.y;
    }*/

    public override float getCarVelocity()
    {
        
        /* 
        // transform objects that velocity on z axis always indicates the direction -> getting the Sign givs the direction
        float direction = Math.Sign(this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity).z);


        // signed speed (foreward and backward speed)
        float velocity = direction * frontLeftWheelCollider.attachedRigidbody.velocity.magnitude;

        if (this.gameObject.transform.name == $"JetBot 0")
        {
            Vector3 plainVelocity = this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity);
            Debug.Log($"getCarVelocity: left {inputAccelerationLeft} right {inputAccelerationRight} direction {direction} velocity {velocity} plainVelocity {plainVelocity} for JetBot 0");
        }*/

        Vector3 plainVelocity = this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity);

        // we only want the velocity along the forward axis:
        float velocity = plainVelocity.z;
        // if we take the magnitude as before we also get the velocity caused by turning on the spot

        return velocity;
    }
}