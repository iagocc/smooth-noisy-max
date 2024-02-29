import numpy as np

from functools import lru_cache

from utils.np_cache import np_cache


@np_cache
def get_difference_highest(x: np.ndarray) -> int:
    highest = x.max()
    second_high = np.partition(x, -2)[-2]
    return highest - second_high


@lru_cache(maxsize=512)
def sensitivity_tree_func(diff: int, t: int) -> int:
    if t >= diff:
        return 1

    return 0


def sum_sensitivity_tree_func(x: np.ndarray, t: int) -> int:
    diff = get_difference_highest(x)
    return np.sum([sensitivity_tree_func(diff, i) for i in range(t)])


def b(x: np.ndarray, i: int) -> int:
    if i == 0:
        return 0

    if i > 0:
        return sum_sensitivity_tree_func(x, i)

    return -sum_sensitivity_tree_func(x, i)


def u(x: np.ndarray, r: int) -> float:
    if np.argmax(x) == r:
        return 1

    return 0


def dampening_function(x: np.ndarray, r: int) -> float:
    i = 0
    while True:
        bi = b(x, i)
        bip = b(x, i + 1)
        score = u(x, r)
        if bi <= score < bip:
            break
        i += 1

    return (score - bi) / (bip - bi) + i
