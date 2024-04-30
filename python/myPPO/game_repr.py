from dataclasses import dataclass

@dataclass(repr=True, eq=True, frozen=True) # unsafe_hash=True)
class GameRepresentation:
    endEvent: str
    collision: bool
    passedFirstGoal: bool
    passedSecondGoal: bool
    passedThirdGoal: bool