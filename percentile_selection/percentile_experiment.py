import numpy as np
import argparse

from mechanims import em_pmf, ld_pmf, rnm_pmf, expected_val, rnm_tdist_pmf, rnm_lap_pmf
from utility import utility, get_percentile_point, calc_smooth_sens, min_t
from utils import bucketize

np.seterr(all="raise")

# Usage example:
# python -u percentile_experiment.py hepth 99 | tee results/hepth_99.dat
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Percentile RLNM experimentation.")
    parser.add_argument(
        "dataset", metavar="dataset", type=str, help="the dataset name."
    )
    parser.add_argument(
        "percentile", metavar="percentile", type=int, help="desired percentile"
    )
    parser.add_argument(
        "-b",
        "--buckets",
        type=int,
        default=0,
        help="equal-width binning process, this represents the amount of bins. If less than one the experiment will not be bucketized.",
    )

    args = parser.parse_args()

    dataset = args.dataset.upper()
    p = args.percentile
    bucket_size = args.buckets

    x = np.load(f"data/1D/{dataset}.n4096.npy").astype(np.float64)
    x.sort()

    if bucket_size > 1:
        x = bucketize(x, bin)

    n = x.size

    u = utility(x, p)
    epss = np.logspace(-1, 2, num=30)
    # epss = np.linspace(0.1, 10, 40)
    m = get_percentile_point(p, n)
    gs = 1
    point_t = min_t(u, p)

    # print("p\teps\tsmooth\terr_em\terr_pf\terr_rlnm\terr_tdist")
    print("p\teps\tsmooth\terr_lap")
    for eps in epss:
        delta = 1e-6
        beta = eps/(2*np.log(2/delta))
        ss = min(calc_smooth_sens(u, beta), gs)  # gs is the upper bound

        # pmf_em = em_pmf(u, gs, eps)
        # pmf_pf = rnm_pmf(u, eps, gs)
        # pmf_ld = ld_pmf(u, gs, eps, p, point_t)
        # pmf_rlnm = rnm_pmf(u, eps, ss)
        # pmf_tdist = rnm_tdist_pmf(u, eps, ss)
        pmf_lap = rnm_lap_pmf(u, eps, ss)

        # err_em = expected_val(u, pmf_em)
        # err_pf = expected_val(u, pmf_pf)
        # err_ld = expected_error(u, pmf_ld)
        # err_rlnm = expected_val(u, pmf_rlnm)
        # err_tdist = expected_val(x, p, pmf_tdist)
        err_lap = expected_val(x, p, pmf_lap)

        print(
            # f"{p}\t{eps:.4f}\t{ss:.6f}\t{err_em:.6f}\t{err_pf:.6f}\t{err_rlnm:.6f}\t{err_tdist:.6f}"
            f"{p}\t{eps:.4f}\t{ss:.6f}\t{err_lap:.6f}"
        )
