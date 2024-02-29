import random
import numpy as np


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
