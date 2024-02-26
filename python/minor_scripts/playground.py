
success = 1

count = 3

rate = success / count

print(rate)

from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

color_int = 2
color_enum = Color(color_int)

print(color_enum)

from unityGymEnv import MapType

map_type_int = 1
map_type_enum = MapType.getRandomEasy()
print(map_type_enum)