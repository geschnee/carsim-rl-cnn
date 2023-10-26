using UnityEngine;
using System.Collections;

public class Diagnosis : MonoBehaviour
{
    void OnApplicationQuit()
    {
        // TODO find out why the unity editor scene playing stops
        Debug.Log("Application ending after " + Time.time + " seconds");
    }
}