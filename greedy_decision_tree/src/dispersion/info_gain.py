from collections import Counter
from src.dispersion.dispersion import DispersionMeasure
from src.dispersion.entropy import Entropy
from numpy import ndarray
from scipy.special import lambertw

import numpy as np


class InfoGain(DispersionMeasure):
    H: Entropy

    def __init__(self) -> None:
        self.H = Entropy()

    def __call__(self, data: np.ndarray, mask: np.ndarray):
        H_t = self.H(data[:, -1].tolist())

        pl = np.count_nonzero(~mask) / data.shape[0]
        pr = np.count_nonzero(mask) / data.shape[0]

        H_tl = self.H(data[~mask, -1].tolist())
        H_tr = self.H(data[mask, -1].tolist())

        H_st = pl * H_tl + pr * H_tr  # Average entropy of the child nodes
        return H_t - H_st

    def global_sensitivity(self, n: int) -> float:
        return np.log(n + 1) + 1 / np.log(2)

    def local_sensitivity_at(self, n: int, t: int) -> float:
        return np.log(t + n) + t / np.log(2)

    def smooth_sensitivity(self, n: int, eps: int) -> float:
        t_max = np.exp(lambertw((2 * n) / (eps)) / 2).real / n
        sens = np.exp(-t_max * eps) * self.local_sensitivity_at(n, t_max)
        return np.min([sens, self.global_sensitivity(n)])


class InfoGainMultiple(InfoGain):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: ndarray, attr: int, vals: set):
        H_t = self.H(data[:, -1].tolist())

        H_ta = 0
        for v in vals:
            mask = data[:, attr] == v
            p_a = np.count_nonzero(mask) / data.shape[0]
            H_ta += p_a * self.H(data[mask, -1].tolist())

        return H_t - H_ta


class InfoGainFriedman(InfoGain):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: np.ndarray, attr: int) -> float:
        ig = 0
        for j, tj in Counter(data[:, attr]).items():
            mask = data[:, attr] == j
            tjcs = np.array(list(Counter(data[mask, -1]).values()))
            ig += float(np.sum(tjcs * np.log2(tjcs / tj)))

        return ig
