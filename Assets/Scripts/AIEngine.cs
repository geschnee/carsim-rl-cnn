using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIEngine : AIEngineBase
{


    private float currentSteerAngle;

    private float maxTorque = 100f;
    private float maxSteeringAngle = 75f;

    public WheelCollider frontLeftWheelCollider;
    public WheelCollider frontRightWheelCollider;
    public WheelCollider rearLeftWheelCollider;
    public WheelCollider rearRightWheelCollider;

    public Transform frontLeftWheelTransform;
    public Transform frontRightWheeTransform;
    public Transform rearLeftWheelTransform;
    public Transform rearRightWheelTransform;


    public override void HandleMotor()
    {


        // Calculate steering angle for each wheel based on difference in acceleration
        float accelerationDiff = Math.Abs(this.inputAccelerationRight) - Math.Abs(this.inputAccelerationLeft);
        float steeringAngle = maxSteeringAngle * accelerationDiff;

        // Apply differential torque to the wheels based on steering angle
        //leftTorque *= 1 - differentialFactor * Mathf.Abs(steeringAngle);
        //rightTorque *= 1 + differentialFactor * Mathf.Abs(steeringAngle);

        // Apply torque and steering angle to the left wheel
        // float leftTorque = (inputAccelerationLeft * this.maxTorque);
        // Maximilan's code applied different torques to the two wheels

        float torque = (inputAccelerationLeft + inputAccelerationRight) / 2;
        torque = (torque * this.maxTorque);

        frontLeftWheelCollider.motorTorque = torque; // leftTorque;
        frontLeftWheelCollider.steerAngle = steeringAngle;

        //Debug.Log($"steering angle {steeringAngle} left {frontLeftWheelCollider.steerAngle} right {frontRightWheelCollider.steerAngle}");
        //Debug.Log($"Motor torque left {frontLeftWheelCollider.motorTorque} right {frontRightWheelCollider.motorTorque}");

        // Apply torque and steering angle to the right wheel
        // float rightTorque = (inputAccelerationRight * this.maxTorque);
        frontRightWheelCollider.motorTorque = torque; //rightTorque;
        frontRightWheelCollider.steerAngle = steeringAngle;

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
        UpdateSingleWheel(rearRightWheelCollider, rearRightWheelTransform);
        UpdateSingleWheel(rearLeftWheelCollider, rearLeftWheelTransform);

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

        return velocity;
    }
}
