using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIEngineClassicalSteering : AIEngineBase
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
        // code by maximilian applied different torques to the two wheels based on the (calculated) steering angle

        // Calculate steering angle for each wheel based on difference in acceleration
        float accelerationDiff = Math.Abs(this.inputAccelerationRight) - Math.Abs(this.inputAccelerationLeft);
        float steeringAngle = maxSteeringAngle * accelerationDiff;


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
    }


    public override float getCarVelocity()
    {

        Vector3 plainVelocity = this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity);

        // we only want the velocity along the forward axis:
        float velocity = plainVelocity.z;

        return velocity;
    }
}
