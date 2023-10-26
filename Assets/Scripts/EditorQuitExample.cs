using UnityEngine;
using UnityEditor;

// Ensure class initializer is called whenever scripts recompile
[InitializeOnLoad]
public class EditorQuitExample
{
    static void Quit()
    {
        Debug.LogWarning("Quitting the Editor");
    }

    static EditorQuitExample()
    {
        EditorApplication.quitting += Quit;
    }
}