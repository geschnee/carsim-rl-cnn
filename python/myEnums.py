import numpy as np

from enum import Enum
class Spawn(Enum):
    Fixed = 0
    OrientationRandom = 1
    OrientationVeryRandom = 2
    FullyRandom = 3


class MapType(Enum):
    random = 0
    easyGoalLaneMiddleBlueFirst = 1
    easyGoalLaneMiddleRedFirst = 2

    twoGoalLanesBlueFirstLeftMedium = 3
    twoGoalLanesBlueFirstRightMedium = 4
    twoGoalLanesRedFirstLeftMedium = 5
    twoGoalLanesRedFirstRightMedium = 6

    twoGoalLanesBlueFirstLeftHard = 7
    twoGoalLanesBlueFirstRightHard = 8
    twoGoalLanesRedFirstLeftHard = 9
    twoGoalLanesRedFirstRightHard = 10

    # pseudo enums types
    randomEvalEasy = 11 
    randomEvalMedium = 12
    randomEvalHard = 13
    randomEval=14
    randomEvalEasyOrMedium = 15

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
            return myEnum.getRandomEasyOrMedium()
        else:
            # pseudoEnum is not a pseudo enum (real enum)
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
    def getRandomEasyOrMedium(myEnum):
        return MapType(np.random.choice([1,2,3,4,5,6]))
    

class EndEvent(Enum):
    NotEnded = 0
    Success = 1
    OutOfTime = 2
    WallHit = 3
    GoalMissed = 4
    RedObstacle = 5
    BlueObstacle = 6