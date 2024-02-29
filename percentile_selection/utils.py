import numpy as np

from utility import percentile


def bucketize(x, nbins):
    x_min, x_max = np.min(x), np.max(x)
    buckets = np.linspace(x_min, x_max, num=nbins)
    return np.digitize(x, buckets)


def mae(x, p, reported):
    actual = percentile(x, p)
    return np.abs(actual - x[reported])


def run(mec, x, u, sens, eps, p, times=1000):
    errors = []
    for _ in range(times):
        errors.append(mae(x, p, mec(u, sens, eps)))

    return np.mean(errors), np.std(errors)
