using UnityEngine;

public enum CollisionMode
{
    unrestricted = 0,
    oncePerTimestep = 1,
    oncePerEpisode = 2,
    terminate = 3,
    ignoreCollisions = 4,
}

public enum EpisodeStatus
{
    Running = 0,
    Success = 1,
    Timeout = 2,
    Collision = 3,
    FinishWithoutAllGoals = 4,
    WaitingForStep = 5,
}

public enum LightSetting
{
    random = 0,
    bright = 1,
    standard = 2,
    dark = 3,
}

public enum SpawnOrientation
{
    Fixed = 0,
    OrientationRandom = 1,
    OrientationVeryRandom = 2,
    //FullyRandom = 3, //deprecated
}

public enum MapType
{
    random = 0,
    easyBlueFirst = 1,
    easyRedFirst = 2,

    mediumBlueFirstLeft = 3,
    mediumBlueFirstRight = 4,
    mediumRedFirstLeft = 5,
    mediumRedFirstRight = 6,

    hardBlueFirstLeft = 7,
    hardBlueFirstRight = 8,
    hardRedFirstLeft = 9,
    hardRedFirstRight = 10,
}

static class MapTypeMethods
{

    public static string GetDifficulty(this MapType mt)
    {
        if ((int) mt == 0)
        {
            return "random";
        }
        else if ((int) mt == 1 || (int) mt == 2)
        {
            return "easy";
        }
        else if ((int) mt > 2 && (int) mt < 7)
        {
            return "medium";
        }
        else if ((int) mt > 6 && (int) mt < 11)
        {
            return "hard";
        }
        else
        {
            Debug.LogError("Error: MapType not recognized");
            return "Error: MapType not recognized";
        }
    }
}