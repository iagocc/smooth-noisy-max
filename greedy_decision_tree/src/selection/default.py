from src.selection.selection import Selection

import numpy as np


class DefaultSelection(Selection):
    def __call__(self, u: np.ndarray) -> int:
        return np.argmax(u)
