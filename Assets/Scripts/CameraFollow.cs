using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    // position that the camera has relativ to the car, the whole time
    public Vector3 cameraPositionRelativeToCar;
    private Transform carBodyTransform;

    private void Awake()
    {
        // assign the carBody transform of the assigned car to the camera script
        this.carBodyTransform = transform.parent.Find("carBody");

        // calculate the camera position relative to the car at the beginning
        // so that the relative camera position can be configured in Unity editor
        cameraPositionRelativeToCar = transform.localPosition - carBodyTransform.localPosition;

        Debug.LogWarning("camera follow script started, cant this script be removed? TODO");
        // this script is attached to the jetbot camera

        // TODO mal das script ausschalten und play_game_from_python machen
    }

    private void FixedUpdate()
    {
        //HandleTranslation();
        //HandleRotation();
    }

    private void HandleTranslation()
    {
        //transform.position = carBodyTransform.TransformPoint(cameraPositionRelativeToCar); //Vector3.Lerp(transform.position, targetPosition, translateSpeed * Time.deltaTime);
        transform.localPosition = cameraPositionRelativeToCar;
        Debug.LogWarning("how was Handle Trnaslation called? TODO can we remove this script?");
    }
    private void HandleRotation()
    {
        transform.rotation = carBodyTransform.rotation;
        Debug.LogWarning("how was Handle rotation called? TODO can we remove this script?");
        // TODO can we remove this script?
    }
}
