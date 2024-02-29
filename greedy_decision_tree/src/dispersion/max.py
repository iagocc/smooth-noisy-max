from collections import Counter
import numpy as np
from src.dispersion.dispersion import DispersionMeasure

class Max(DispersionMeasure):

    def __call__(self, data: np.ndarray, attr: int):
        u = np.zeros(data.shape[1] - 1)

        op = []
        for att in range(data.shape[1] - 1):
            val = 0
            for j, _ in Counter(data[:, att]).items():
                mask = data[:, att] == j
                tjcs = np.array(list(Counter(data[mask, -1]).values()))
                val += tjcs.max()

            op.append(val)

        maxop = np.array(op)
        u[np.argmax(maxop)] = 1
        return u[attr]
    
    def global_sensitivity(self, n: int):
        return 1
    
    def local_sensitivity_at(self, n: int, t: int):
        raise NotImplemented("The dispersion measure sensitivity should be implemented.")

    def smooth_sensitivity(self, data: np.ndarray, eps: float):
        op = []
        for att in range(data.shape[1] - 1):
            val = 0
            for j, _ in Counter(data[:, att]).items():
                mask = data[:, att] == j
                tjcs = np.array(list(Counter(data[mask, -1]).values()))
                val += tjcs.max()

            op.append(val)

        maxop = np.array(op)

        min_t = -1 * np.max((np.sort(maxop)[::-1] - np.max(maxop))[1:])
        return np.max([np.exp(-min_t*eps), 1e-9]) # due to numerical error