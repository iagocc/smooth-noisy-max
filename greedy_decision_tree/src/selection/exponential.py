import numpy as np

from dataclasses import dataclass
from random import choices

import mpmath as mp


@dataclass
class ExponentialMechanism:
    eps: float
    sens: float

    def __str__(self) -> str:
        return "EM"

    def __call__(self, u: np.ndarray) -> int:
        assert self.eps is not None and self.eps > 0, "The budget parameter should be set."

        scaled_u = [((r * self.eps) / (2 * self.sens)) for r in u]

        if sum([abs(u) > 700.0 for u in scaled_u]) > 0:
            exp_u = [mp.exp(s) for s in scaled_u]
            sum_exp = mp.fsum(exp_u)
            scores = np.array([float(mp.fdiv(s, sum_exp)) for s in exp_u])
        else:
            exp_u = np.exp(scaled_u)
            scores = exp_u / exp_u.sum()

        return choices(range(scores.shape[0]), scores, k=1)[0]
