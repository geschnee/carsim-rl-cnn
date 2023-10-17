using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    private const string HORIZONTAL = "Horizontal";
    private const string VERTICAL = "Vertical";


    private float currentSteerAngle;

    [SerializeField] private float motorForce;
    [SerializeField] private float maxSteerAngle;

    [SerializeField] private WheelCollider frontLeftWheelCollider;
    [SerializeField] private WheelCollider frontRightWheelCollider;
    [SerializeField] private WheelCollider rearLeftWheelCollider;
    [SerializeField] private WheelCollider rearRightWheelCollider;

    [SerializeField] private Transform frontLeftWheelTransform;
    [SerializeField] private Transform frontRightWheeTransform;
    [SerializeField] private Transform rearLeftWheelTransform;
    [SerializeField] private Transform rearRightWheelTransform;
    

	[SerializeField] private Transform carBody;
	[SerializeField] public int count;

    private DrivingEngine drivingEngine;
    
    private void Awake()
    {
        this.drivingEngine = new DrivingEngine( this.motorForce, this.maxSteerAngle, this.carBody, this.frontLeftWheelCollider, this.frontRightWheelCollider, this.rearLeftWheelCollider, this.rearRightWheelCollider,
            this.frontLeftWheelTransform, this.frontRightWheeTransform, this.rearLeftWheelTransform, this.rearRightWheelTransform);
    }

    private void FixedUpdate()
    {
        this.drivingEngine.SetInput(this.GetInput());
        this.drivingEngine.HandleMotor();
        this.drivingEngine.HandleSteering();
        this.drivingEngine.UpdateWheels();
    }


    private List<float> GetInput()
    {
        float accelerationInput = Input.GetAxis(HORIZONTAL);
        float steerInput = Input.GetAxis(VERTICAL);
        return new List<float>(){ accelerationInput, steerInput };
    }
    
  
    private float getCarVelocity()
    {
        // transform objects that velocity on z axis always indicates the direction -> getting the Sign givs the direction
        float direction = Math.Sign(carBody.InverseTransformDirection(frontLeftWheelCollider.attachedRigidbody.velocity).z);

        // signed speed (foreward and backward speed)
        float velocity = direction * frontLeftWheelCollider.attachedRigidbody.velocity.magnitude;

        return velocity;

    }

}
