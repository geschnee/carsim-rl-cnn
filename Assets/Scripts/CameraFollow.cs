using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    // position that the camera has relativ to the car, the whole time
    public Vector3 cameraPositionRelativeToCar; // = new Vector3(0, 2, 0.5f);
    private Transform carBodyTransform;

    private void Awake()
    {
        // assign the carBody transform of the assigned car to the camera script
        this.carBodyTransform = transform.parent.Find("carBody");

        // calculate the camera position relative to the car at the beginning
        // so that the relative camera position can be configured in Unity editor
        cameraPositionRelativeToCar = transform.localPosition - carBodyTransform.localPosition;
    }

    private void FixedUpdate()
    {
        HandleTranslation();
        HandleRotation();
    }

    private void HandleTranslation()
    {
        //transform.position = carBodyTransform.TransformPoint(cameraPositionRelativeToCar); //Vector3.Lerp(transform.position, targetPosition, translateSpeed * Time.deltaTime);
        transform.localPosition = cameraPositionRelativeToCar;
    }
    private void HandleRotation()
    {
        transform.rotation = carBodyTransform.rotation;
    }
}
