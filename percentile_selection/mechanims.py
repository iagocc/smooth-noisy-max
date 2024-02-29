import numpy as np
import scipy.integrate as integrate
import operator
import mpmath as mp

from random import choices
from numba import njit


@njit(cache=True, nogil=True, fastmath=True)
def safe_exp(x: float) -> float:
    sign = 1 if x >= 0 else -1
    x = 700 * sign if abs(x) > 700 else x  # clip
    return float(np.exp(x))


# @njit(cache=True, nogil=True, fastmath=True)
# def safe_exp(x: float) -> np.float128:
#     return np.exp(np.array([x]).astype(np.float128))[0]


# def safe_exp(v: float) -> float:
#     try:
#         return float(np.exp(v))
#     except (FloatingPointError, RuntimeWarning):
#         return float(mp.exp(v))


def em_pmf(u: np.ndarray, gs: float, eps: float):
    scaled_u = np.array([safe_exp((r * eps) / (2 * gs)) for r in u])
    scores = scaled_u / scaled_u.sum()
    return scores


def em(u: np.ndarray, gs: float, eps: float):
    scores = em_pmf(u, gs, eps)
    return choices(range(scores.size), scores, k=1)[0]


def rnm(u: np.ndarray, gs: float, eps: float):
    noise = np.random.default_rng().exponential(scale=((2 * gs) / eps), size=u.size)
    return np.argmax(u + noise)


def rlnm(u: np.ndarray, ss: float, eps: float):
    return rnm(u, ss, eps)


@njit(cache=True, nogil=True, fastmath=True)
def cdf(x: float, scale: float) -> float:
    if x < 0:
        return 0
    return 1 - safe_exp(-x / scale)


@njit(cache=True, nogil=True, fastmath=True)
def pdf(x: float, scale: float) -> float:
    if x < 0:
        return 0
    return safe_exp(-x / scale) / scale


@njit
def mul(x, y):
    return operator.mul(x, y)


@njit(cache=True, nogil=True, fastmath=True)
def prod_step(r: int, u: np.ndarray, x: float, n: int, scale: float) -> float:
    prod = 1
    for s in range(n):
        if r == s:
            continue
        else:
            prod *= cdf(u[r] - u[s] + x, scale)
    return prod


@njit(cache=True, nogil=True, fastmath=True)
def int_step(x: float, u: np.ndarray, r: int, scale: float, n: int) -> float:
    fx = pdf(x, scale)
    # items = np.array([cdf(u[r] - u[s] + x, scale) for s in range(n) if r != s])
    # prod = np.prod(items)
    prod = prod_step(r, u, x, n, scale)

    return fx * prod


def get_rnm_probs(n: int, scale: float, u: np.ndarray) -> np.ndarray:
    probs = []
    for r in range(n):
        p, _ = integrate.quad_vec(int_step, 0, np.inf, args=(u, r, scale, n))
        probs.append(p)

    assert np.isclose(
        np.array(probs).sum(), 1.0
    ), f"Probs. should sum one (actual {np.array(probs).sum()}, scale {scale})."
    return np.array(probs)


def rnm_pmf(u: np.ndarray, eps: float, sens: float) -> np.ndarray:
    n: int = u.size
    scale: float = (2 * sens) / eps
    return get_rnm_probs(n, scale, u)


def expected_error(u, pmf):
    return u.max() - np.sum(u @ pmf)


def expected_error_val(x, k, pmf):
    return np.sum(x @ pmf)


def abs_error(gt: float, pred: float) -> float:
    return np.abs(gt - pred)


@njit(cache=True, nogil=True, fastmath=True)
def dampening_func(u: np.ndarray, r: int, gs: float, p: int, min_t: int) -> float:
    sign = 1
    if u[r] < 0:
        sign = -1
        u[r] *= sign

    t = 0
    bt = b(u, t, r, min_t)
    while True:
        btp = b(u, t + 1, r, min_t)

        if btp - bt >= gs - 1e-6 or btp - bt == 0:
            return ((u[r] - bt) / gs) * sign

        if u[r] >= bt and u[r] < btp:
            break

        t += 1

    return (((u[r] - bt) / (btp - bt)) + t) * sign


@njit(cache=True, nogil=True, fastmath=True)
def b(u: np.ndarray, t: int, r: int, min_t: int) -> int:
    if t == 0 or t < min_t:
        return 0

    return t - min_t


def ld_pmf(u: np.ndarray, gs: float, eps: float, p: int, min_t: int):
    u = np.array(u, dtype=np.int_)
    dampened_u = np.array([dampening_func(u, el, gs, p, min_t) for el in range(u.size)])
    return em_pmf(dampened_u, gs, eps)


def ld(u: np.ndarray, gs: float, eps: float, p: int, min_t: int):
    scores = ld_pmf(u, gs, eps, p, min_t)
    return choices(range(scores.shape[0]), scores, k=1)[0]
