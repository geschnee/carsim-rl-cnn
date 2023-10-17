using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrivingEngine
{

    private float inputAcceleration;
    private float inputSteering;
    private float currentSteerAngle;

    private float motorForce;
    private float maxSteerAngle;

    private WheelCollider frontLeftWheelCollider;
    private WheelCollider frontRightWheelCollider;
    private WheelCollider rearLeftWheelCollider;
    private WheelCollider rearRightWheelCollider;

    private Transform frontLeftWheelTransform;
    private Transform frontRightWheeTransform;
    private Transform rearLeftWheelTransform;
    private Transform rearRightWheelTransform;


    private Transform carBody;

    public DrivingEngine(float motorForce, float maxSteerAngle, Transform carBody, WheelCollider frontLeftWheelCollider, WheelCollider frontRightWheelCollider, WheelCollider rearLeftWheelCollider, WheelCollider rearRightWheelCollider,
        Transform frontLeftWheelTransform, Transform frontRightWheeTransform, Transform rearLeftWheelTransform, Transform rearRightWheelTransform)
    {

        this.motorForce = motorForce;
        this.maxSteerAngle = maxSteerAngle;

        this.carBody = carBody;

        this.frontLeftWheelCollider = frontLeftWheelCollider;
        this.frontRightWheelCollider = frontRightWheelCollider;
        this.rearLeftWheelCollider = rearLeftWheelCollider;
        this.rearRightWheelCollider = rearRightWheelCollider;

        this.frontLeftWheelTransform = frontLeftWheelTransform;
        this.frontRightWheeTransform = frontRightWheeTransform;
        this.rearLeftWheelTransform = rearLeftWheelTransform;
        this.rearRightWheelTransform = rearRightWheelTransform;

}

    public void SetInput(List<float> input)
    {
        this.inputAcceleration = input[0];
        this.inputSteering = input[1];
    }

    public void HandleMotor()
    {
        //resistance slows down car when not accelarating
        // grows with velocity + (signed in direction of vel) constant

        float resistance = (this.getCarVelocity() * 10f) + Math.Sign(this.getCarVelocity()) * 15f;


        frontLeftWheelCollider.motorTorque = (inputSteering * motorForce) - resistance;
        frontRightWheelCollider.motorTorque = (inputSteering * motorForce) - resistance;

    }

    public void HandleSteering()
    {
        currentSteerAngle = maxSteerAngle * inputAcceleration;
        frontLeftWheelCollider.steerAngle = currentSteerAngle;
        frontRightWheelCollider.steerAngle = currentSteerAngle;
    }

    public void UpdateWheels()
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
        // Debug.Log(pos);
        // (25, 87) , (-30, 90) (-30, (-25) , (24, -27)

    }


    public float getCarVelocity()
    {
        // transform objects that velocity on z axis always indicates the direction -> getting the Sign givs the direction
        float direction = Math.Sign(this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity).z);

        // signed speed (foreward and backward speed)
        float velocity = direction * frontLeftWheelCollider.attachedRigidbody.velocity.magnitude;

        return velocity;

    }

}
