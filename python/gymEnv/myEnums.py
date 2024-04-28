import numpy as np

from enum import Enum

class CollisionMode(Enum):
    unrestricted = 0
    oncePerTimestep = 1
    oncePerEpisode = 2
    resetUponCollision = 3
    ignoreCollisions = 4

class Spawn(Enum):
    Fixed = 0
    OrientationRandom = 1
    OrientationVeryRandom = 2
    FullyRandom = 3

class LightSetting(Enum):
    random = 0
    bright = 1
    standard = 2
    dark = 3

    @classmethod
    def resolvePseudoEnum(myEnum, pseudoEnum):
        if pseudoEnum.value == 0:
            return myEnum.getRandom()
        else:
            return pseudoEnum
    
    @classmethod
    def getRandom(myEnum):
        return LightSetting(np.random.choice([1,2,3]))

class MapType(Enum):
    random = 0
    easyBlueFirst = 1
    easyRedFirst = 2

    mediumBlueFirstLeft = 3
    mediumBlueFirstRight = 4
    mediumRedFirstLeft = 5
    mediumRedFirstRight = 6

    hardBlueFirstLeft = 7
    hardBlueFirstRight = 8
    hardRedFirstLeft = 9
    hardRedFirstRight = 10

    # pseudo enums types
    randomEvalEasy = 11 
    randomEvalMedium = 12
    randomEvalHard = 13
    randomEval=14
    randomEvalIncreasedMediumAndHard = 15

    @classmethod
    def resolvePseudoEnum(myEnum, pseudoEnum):
        if pseudoEnum.value == 0:
            return myEnum.getRandomEval()
        elif pseudoEnum.value == 11:
            return myEnum.getRandomEasy()
        elif pseudoEnum.value == 12:
            return myEnum.getRandomMedium()
        elif pseudoEnum.value == 13:
            return myEnum.getRandomHard()
        elif pseudoEnum.value == 14:
            return myEnum.getRandomEval()
        elif pseudoEnum.value == 15:
            return myEnum.getRandomIncreasedMediumAndHard()
        else:
            # pseudoEnum is not a pseudo enum (it is real enum)
            return pseudoEnum

    @classmethod
    def getRandomEasy(myEnum):
        return MapType(np.random.choice([1,2]))
    
    @classmethod
    def getRandomMedium(myEnum):
        return MapType(np.random.choice([3,4,5,6]))
    
    @classmethod
    def getRandomHard(myEnum):
        return MapType(np.random.choice([7,8,9,10]))
    
    @classmethod
    def getMapTypeFromDifficulty(myEnum, difficulty):
        if difficulty == "easy":
            return myEnum.getRandomEasy()
        elif difficulty == "medium":
            return myEnum.getRandomMedium()
        elif difficulty == "hard":
            return myEnum.getRandomHard()
        else:
            assert False, f'unknown difficulty {difficulty}'

    @classmethod
    def getRandomEval(myEnum):
        return MapType(np.random.choice([1,2,3,4,5,6,7,8,9,10]))

    @classmethod
    def getRandomIncreasedMediumAndHard(myEnum):
        # random eval is 20% easy, 40% medium, 40% hard

        # this is 10% easy, 45% medium, 45% hard

        difficulty = np.random.choice([0,1,2], p=[0.1, 0.45, 0.45])
        if difficulty == 0:
            return myEnum.getRandomEasy()
        elif difficulty == 1:
            return myEnum.getRandomMedium()
        elif difficulty == 2:
            return myEnum.getRandomHard()
        
    

class EndEvent(Enum):
    NotEnded = 0
    Success = 1
    OutOfTime = 2
    WallHit = 3
    GoalMissed = 4
    RedObstacle = 5
    BlueObstacle = 6