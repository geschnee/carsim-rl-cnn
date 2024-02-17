using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using Unity.MLAgents;
//using Unity.MLAgents.Actuators;
//using Unity.MLAgents.Sensors;
using System;
using System.IO;
using System.Linq;


public class BallColor : MonoBehaviour
{
    public Material redMaterial;
    public Material greenMaterial;

    public void SetRed()
    {
        GetComponent<Renderer>().material = redMaterial;
    }

    public void SetGreen()
    {
        GetComponent<Renderer>().material = greenMaterial;
    }
}
