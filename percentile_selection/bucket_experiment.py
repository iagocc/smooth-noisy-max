import numpy as np
import argparse

from mechanims import em_pmf, rnm_pmf, expected_error
from utility import utility, get_percentile_point, calc_smooth_sens
from utils import bucketize

np.seterr(all="raise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Percentile RLNM experimentation.")
    parser.add_argument(
        "dataset", metavar="dataset", type=str, help="the dataset name."
    )
    parser.add_argument(
        "percentile", metavar="percentile", type=int, help="desired percentile"
    )
    parser.add_argument("eps", metavar="eps", type=float, help="privacy budget param.")
    parser.add_argument(
        "bucket_size_lower",
        metavar="bucket_size_lower",
        type=int,
        help="the lower value of bucket size.",
    )
    parser.add_argument(
        "bucket_size_upper",
        metavar="bucket_size_upper",
        type=int,
        help="the upper value of bucket size.",
    )
    parser.add_argument(
        "bucket_size",
        metavar="bucket_size",
        type=int,
        help="the amount of bucket size to be testes.",
    )

    args = parser.parse_args()

    dataset = args.dataset.upper()
    p = args.percentile
    lower, upper = args.bucket_size_lower, args.bucket_size_upper
    bsize = args.bucket_size
    eps = args.eps

    print("eps\tbsize\tsmooth\terr_rlnm")
    for b in np.linspace(lower, upper, bsize, endpoint=True, dtype=np.int_):
        x = np.load(f"data/1D/{dataset}.n4096.npy").astype(np.float64)
        x = bucketize(x, b)
        x.sort()

        n = x.size

        u = utility(x)
        m = get_percentile_point(p, n)
        gs = 1

        ss = min(calc_smooth_sens(u, eps), gs)  # gs is the upper bound

        pmf_rlnm = rnm_pmf(u, eps, ss)

        err_rlnm = expected_error(u, pmf_rlnm)

        print(f"{eps:.4f}\t{b}\t{ss:.6f}\t{err_rlnm:.6f}")
