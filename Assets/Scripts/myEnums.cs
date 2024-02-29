

public enum EpisodeStatus
{
    Running = 0,
    Success = 1,
    OutOfTime = 2,
    WallHit = 3,
    GoalMissed = 4,
    RedObstacle = 5,
    BlueObstacle = 6,
    FinishMissed = 7,
    WaitingForStep = 8,
}

public enum LightSetting
{
    random = 0,
    bright = 1,
    standard = 2,
    dark = 3,
}

public enum Spawn
{
    Fixed = 0,
    OrientationRandom = 1,
    OrientationVeryRandom = 2,
    FullyRandom = 3,
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