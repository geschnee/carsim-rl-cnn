using UnityEngine;
using UnityEditor;

// Ensure class initializer is called whenever scripts recompile
[InitializeOnLoad]
public class EditorWantsToQuitExample
{
    static bool WantsToQuit()
    {
        Debug.Log("Editor prevented from quitting.");
        return false;
    }

    static EditorWantsToQuitExample()
    {
        EditorApplication.wantsToQuit += WantsToQuit;
    }
}