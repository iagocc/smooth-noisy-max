from typing import Optional
import numpy as np

from src.selection.selection import Selection
from src.selection.exponential import ExponentialMechanism
from src.selection.dampening.dampening_func import DampeningFunc
from src.selection.dampening.delta_info_gain import DeltaInfoGain

from dataclasses import dataclass


@dataclass
class LocalDampeningMechanism(Selection):
    eps: float
    data: np.ndarray
    shifted: Optional[bool]
    sens: Optional[float]

    def __call__(self, u: np.ndarray) -> int:
        if self.shifted:
            if not self.sens:
                raise ValueError("When the shifted activate the global sensitivity should be available.")

            u = u - (self.sens * self.data.shape[0] + np.max(u))

        C = np.unique(self.data[:, -1])
        D = []
        for o in range(u.shape[0]):
            delta = DeltaInfoGain(self.data, o, C, {})
            damp = DampeningFunc(self.data, delta)
            D.append(damp.evaluate(np.copy(u), o))

        return ExponentialMechanism(eps=self.eps, sens=1)(D)
