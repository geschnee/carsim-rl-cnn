import numpy as np

from enum import Enum

class CollisionMode(Enum):
    unrestricted = 0
    oncePerTimestep = 1
    oncePerEpisode = 2
    terminate = 3
    ignoreCollisions = 4

class SpawnOrientation(Enum):
    Fixed = 0
    Random = 1
    VeryRandom = 2
    # FullyRandom = 3
    # we deprecate FullyRandom because it is not used in the current implementation
    # easier to explain the spawn positions if there are only 3 options

    @classmethod
    def getOrientationRange(myEnum, spawnEnum):
        # returns the interval of spawn rotations corresponding to the Spawn method
        if spawnEnum.value == 0:
            return 0.0, 0.0
        elif spawnEnum.value == 1:
            return -15.0, 15.0
        elif spawnEnum.value == 2:
            return -45.0, 45.0
        elif spawnEnum.value == 3:
            assert False, f'FullyRandom is deprecated'
            return -45, 45
        else:
            assert False, f'unknown spawnEnum {spawnEnum}'

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
    random = 0 # deprecated
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
    def getAllTracknumbersOfDifficulty(myEnum, difficulty):
        if difficulty == "easy":
            return [1,2]
        elif difficulty == "medium":
            return [3,4,5,6]
        elif difficulty == "hard":
            return [7,8,9,10]
        else:
            assert False, f'unknown difficulty {difficulty}'

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
    Running = 0
    Success = 1
    Timeout = 2
    Collision = 3
    FinishWithoutAllGoals = 4
    WaitingForStep = 5
