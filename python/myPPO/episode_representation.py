from dataclasses import dataclass

@dataclass(repr=True, eq=True, frozen=False, unsafe_hash=True)
class EpisodeRepresentation:
    endEvent: str
    collision: bool
    passedFirstGoal: bool
    passedSecondGoal: bool
    passedThirdGoal: bool

    def __init__(self, info):

        self.endEvent = info["endEvent"]
        self.collision = int(info["collision"]) == 1
        self.passedFirstGoal = int(info["passedFirstGoal"]) == 1
        self.passedSecondGoal = int(info["passedSecondGoal"]) == 1
        self.passedThirdGoal = int(info["passedThirdGoal"]) == 1