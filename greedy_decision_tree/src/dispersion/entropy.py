from src.dispersion.dispersion import DispersionMeasure

import collections
import math


class Entropy(DispersionMeasure):
    def __call__(self, data: list):
        probabilities = [n_x / len(data) for _, n_x in collections.Counter(data).items()]
        e_x = [-p_x * math.log(p_x, 2) for p_x in probabilities]
        return sum(e_x)
