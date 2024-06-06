using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIEngineClassicalSteering : AIEngineBase
{

    private float currentSteerAngle;

    private float maxTorque = 200f; // was 100, this agent needs more speed, it reaches timeouts in jetbot generalization test
    private float maxSteeringAngle = 75f;

    public WheelCollider frontLeftWheelCollider;
    public WheelCollider frontRightWheelCollider;
    public WheelCollider rearLeftWheelCollider;
    public WheelCollider rearRightWheelCollider;

    public Transform frontLeftWheelTransform;
    public Transform frontRightWheelTransform;
    public Transform rearLeftWheelTransform;
    public Transform rearRightWheelTransform;


    public override void HandleMotor()
    {
        // code by maximilian applied different torques to the two wheels based on the (calculated) steering angle

        // Calculate steering angle for each wheel based on difference in acceleration
        float accelerationDiff = Math.Abs(this.inputAccelerationLeft) - Math.Abs(this.inputAccelerationRight);
        // was abs(rightAcceleration) - abs(leftAcceleration) before
        // this was switched to be more intuitive
        // higher inputLeftAcceleration means more steering to the left

        float steeringAngle = maxSteeringAngle * accelerationDiff;

        //Debug.Log($"right {this.inputAccelerationRight} left {this.inputAccelerationLeft} Steering angle: " + steeringAngle);


        float torque = (inputAccelerationLeft + inputAccelerationRight) / 2;
        torque = (torque * this.maxTorque);

        frontLeftWheelCollider.motorTorque = torque;
        frontLeftWheelCollider.steerAngle = steeringAngle;

        frontRightWheelCollider.motorTorque = torque;
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
        UpdateSingleWheel(frontRightWheelCollider, frontRightWheelTransform);
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
    }


    public override float getCarVelocity()
    {

        Vector3 plainVelocity = this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity);

        // we only want the velocity along the forward axis:
        float velocity = plainVelocity.z;

        return velocity;
    }
}
