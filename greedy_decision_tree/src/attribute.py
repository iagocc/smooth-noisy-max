from enum import Enum
from typing import Union

AttrVal = Union[str, int, float]


class AttrType(Enum):
    D = "d"
    C = "c"
