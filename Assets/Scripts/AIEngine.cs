using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIEngine : MonoBehaviour
{

    public float inputAccelerationLeft = 0;
    public float inputAccelerationRight = 0;
    private float currentSteerAngle;

    private float maxTorque = 50f; // was 100f;
    // TODO we are trying a smaller value to have less movement per step
    private float maxSteeringAngle = 75f;
    private float resistanceFactor = 1f;

    public WheelCollider frontLeftWheelCollider;
    public WheelCollider frontRightWheelCollider;
    public WheelCollider rearLeftWheelCollider;
    public WheelCollider rearRightWheelCollider;

    public Transform frontLeftWheelTransform;
    public Transform frontRightWheeTransform;
    public Transform rearLeftWheelTransform;
    public Transform rearRightWheelTransform;

    public bool episodeRunning = false;

    public Transform carBody;

    public void Start()
    {
        //Debug.Log($"AIEngine started");
        //Debug.Log($"left front wheeel rotation {frontLeftWheelCollider.transform.rotation.eulerAngles}");
        //Debug.Log($"Motor torque start left {frontLeftWheelCollider.motorTorque} right {frontRightWheelCollider.motorTorque}");

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

    public void HandleMotor()
    {
        //resistance slows down car when not accelarating
        // grows with velocity + (signed in direction of vel) constant
        // custome resistance if wanted actually set to 0



        // this resistance is not used anywhere
        float resistance = (float)(0f * (Math.Pow(this.getCarVelocity(), 8) * Math.Sign(this.getCarVelocity()) * this.resistanceFactor) + Math.Sign(this.getCarVelocity()) * 5f);


        //frontLeftWheelCollider.motorTorque = (inputAccelerationLeft * motorForce) - resistance;
        //frontRightWheelCollider.motorTorque = (inputAccelerationRight * motorForce) - resistance;


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
        float rightTorque = (inputAccelerationRight * this.maxTorque);
        frontRightWheelCollider.motorTorque = torque; //rightTorque;
        frontRightWheelCollider.steerAngle = steeringAngle;

        // TODO why do right and left wheels have different torque?
        // see unity tutorial: https://docs.unity3d.com/2017.4/Documentation/Manual/WheelColliderTutorial.html
        // simply because there were two input values?
        // we could use min, max or mean to get the same torque for both wheels


        //if (inputAccelerationLeft == 0)
        //{
        //    Debug.Log($"leftTorque {leftTorque} inputAccelerationLeft {inputAccelerationLeft} rightTorque {rightTorque} inputAccelerationRight {inputAccelerationRight}");
        //}

    }

    public void ResetMotor()
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
        // TODO why do the wheels (mesh) rotate when the JetBot is fresh?
        // there has not been any command issued yet

        // sieht aus wie dieses Tutorial:
        // https://www.youtube.com/watch?v=rdl66506RY4&list=PL1R2qsKCcUCIdGXBLkZV2Tq_sxa-ADASN

    }

    public float getSteeringAngle()
    {
        //TODO check if correct angle gives back 90° should have 0
        return this.carBody.eulerAngles.y;
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
