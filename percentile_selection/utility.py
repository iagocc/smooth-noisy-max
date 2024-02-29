import numpy as np

from mechanims import safe_exp


def get_percentile_point(p, n):
    return int(np.floor(p * (n) / 100))


def percentile(x, p):
    n = x.size
    i = get_percentile_point(p, n)
    return x[i]


def utility(x, p=50) -> np.ndarray:
    n = x.size
    m = get_percentile_point(p, n)
    m_val = x[m]
    mask = x == m_val
    u = np.zeros(n)
    u[mask] = 1
    return u


def calc_smooth_sens(u, epsilon, p=50, sens_error_min=10e-3):
    t = min_t(u, p)
    return max(float(safe_exp(-t * epsilon)), sens_error_min)


def min_t(u, p=50):
    n = u.size
    m = get_percentile_point(p, n)
    lower_count = np.count_nonzero(u[:m])
    upper_count = np.count_nonzero(u[m + 1 :])
    min_dist = min(lower_count, upper_count)
    return 1 + 2 * min_dist
