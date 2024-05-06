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
    }

    public override float getCarVelocity()
    {

        Vector3 plainVelocity = this.carBody.InverseTransformDirection(this.frontLeftWheelCollider.attachedRigidbody.velocity);

        // we only want the velocity along the forward axis:
        float velocity = plainVelocity.z;
        // if we take the magnitude as before we also get the velocity caused by turning on the spot

        return velocity;
    }
}