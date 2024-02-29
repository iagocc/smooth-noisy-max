import numpy as np
import argparse

from mechanims import em, ld, rnm, rlnm, abs_error
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
    m = get_percentile_point(p, n)
    gs = 1
    point_t = min_t(u, p)
    times = 1

    ground_truth = x[m]

    print("p\teps\tsmooth\terr_em\terr_pf\terr_ld\terr_rlnm")
    for eps in epss:
        ss = min(calc_smooth_sens(u, eps), gs)  # gs is the upper bound

        em_errs = []
        pf_errs = []
        ld_errs = []
        rlnm_errs = []

        for _ in range(times):
            pred_em = em(u, gs, eps)
            pred_pf = rnm(u, gs, eps)
            pred_ld = ld(u, gs, eps, p, point_t)
            pred_rlnm = rlnm(u, ss, eps)

            em_errs.append(abs_error(ground_truth, x[pred_em]))
            pf_errs.append(abs_error(ground_truth, x[pred_pf]))
            ld_errs.append(abs_error(ground_truth, x[pred_ld]))
            rlnm_errs.append(abs_error(ground_truth, x[pred_rlnm]))

        err_em = np.mean(em_errs)
        err_pf = np.mean(pf_errs)
        err_ld = np.mean(ld_errs)
        err_rlnm = np.mean(rlnm_errs)
        print(
            f"{p}\t{eps:.4f}\t{ss:.6f}\t{err_em:.6f}\t{err_pf:.6f}\t{err_ld:.6f}\t{err_rlnm:.6f}"
        )
