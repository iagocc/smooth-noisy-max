from dataclasses import dataclass

import numpy as np

from src.selection.selection import Selection


@dataclass
class RNM(Selection):
    eps: float
    sens: float

    def __call__(self, u: np.ndarray) -> int:
        noise = np.random.default_rng().exponential(scale=((2 * self.sens) / self.eps), size=u.size)
        return int(np.argmax(u + noise))


class RLNM(RNM):
    def __call__(self, u: np.ndarray) -> int:
        return super().__call__(u)
