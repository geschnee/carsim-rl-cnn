using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    // position that the camera has relativ to the car, the whole time
    private Vector3 cameraPositionRelativeToCar = new Vector3(0, 2, 0.5f);
    private Transform carBodyTransform;

    private void Awake()
    {
        // assign the carBody transform of the assigned car to the camera script
        this.carBodyTransform = transform.parent.Find("carBody");
    }

    private void FixedUpdate()
    {
        HandleTranslation();
        HandleRotation();
    }
	
    private void HandleTranslation()
    {
        transform.position = carBodyTransform.TransformPoint(cameraPositionRelativeToCar); //Vector3.Lerp(transform.position, targetPosition, translateSpeed * Time.deltaTime);
    }
    private void HandleRotation()
    {
        transform.rotation = carBodyTransform.rotation;
    }
}
