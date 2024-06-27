from src.dispersion.dispersion import DispersionMeasure
from src.dispersion.info_gain import InfoGainFriedman
from src.dispersion.max import Max
from src.selection.selection import Selection
from src.selection.exponential import ExponentialMechanism
from src.selection.rnm import RNM, RLNM, TStudentRLNM, LlnRLNM, SmoothLaplaceRLNM
from src.selection.local_dampening import LocalDampeningMechanism
from src.dp_id3 import DpID3
from src.dataset import Dataset
from src.logger import TreeLogger

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from copy import deepcopy
from time import process_time_ns as timer

import random
import numpy as np


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def experiment_dpid3_step(
    step: int,
    logger: TreeLogger,
    method: Selection,
    dm: InfoGainFriedman,
    eps: float,
    folds: int,
    depth: int,
    ds: np.ndarray,
    ds_struct: np.ndarray,
    ds_dom: dict,
    ds_name: str,
    eps_total: float,
):
    kf = KFold(n_splits=folds, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(ds)):
        train, test = np.copy(ds[train_index, :]), np.copy(ds[test_index, :])
        tree = DpID3(
            root=None,
            dm=dm,
            selection=method,
            dataset=train,
            dataset_struct=ds_struct,
            dataset_domain=ds_dom,
            max_depth=depth,
            min_sample=10,
            eps=eps,
        )

        start = timer()
        tree.fit(train)
        fit_time = timer() - start

        y_pred = tree.predict(test)
        acc = accuracy_score(test[:, -1].tolist(), y_pred.tolist())

        logger.log(
            dataset=ds_name,
            method=method.__class__.__name__,
            trial=step + i,
            dm=dm.__class__.__name__,
            sens=method.sens,
            eps=eps_total,
            depth=depth,
            acc=float(acc),
            train_time_ns=fit_time,
        )

    logger.save()


def experiment_dpid3():
    logger = TreeLogger(experiment_name="new_ones", verbose=True)
    datasets = [("adult_clean", "income"), ("nltcs", "15"), ("acs", "22")]
    # datasets = [("adult_disc_20bucks", "14")]
    eps = [0.01, 0.05, 0.1, 0.5, 1, 2]
    times = 5
    folds = 10
    depths = [2, 5]
    discritize = 20

    dm = Max()
    max_n = 50_000
    gs = dm.global_sensitivity(max_n)
    delta = 1e-6

    for d in datasets:
        ds, ds_struc = Dataset.load_dataset(d[0], d[1])
        ds, ds_struc = Dataset.discretize_dataset(ds, ds_struc, discritize)
        ds_dom = Dataset.dataset_domain(ds)

        n = ds.shape[0]

        for depth in depths:
            for e in eps:
                privacy_budget = e / (2 * (depth + 1))
                methods = [
                    # ExponentialMechanism(eps=privacy_budget, sens=gs),
                    # RNM(eps=privacy_budget, sens=gs),
                    # LocalDampeningMechanism(eps=deepcopy(privacy_budget), data=np.copy(ds), shifted=True, sens=gs),
                    # RLNM(eps=privacy_budget, sens=dm.smooth_sensitivity(data=ds, eps=e)),
                    TStudentRLNM(eps=privacy_budget, sens=dm.smooth_sensitivity(ds, TStudentRLNM.get_beta(e, 3)), beta=TStudentRLNM.get_beta(e, 3)),
                    LlnRLNM(eps=privacy_budget, sens=dm.smooth_sensitivity(ds, LlnRLNM.get_beta(e)), beta=TStudentRLNM.get_beta(e, 3)),
                    SmoothLaplaceRLNM(eps=privacy_budget, sens=dm.smooth_sensitivity(ds, SmoothLaplaceRLNM.get_beta(e, delta)), delta=delta)
                ]
                for m in methods:
                    for step in range(times):
                        experiment_dpid3_step(
                            step=step * folds,
                            logger=logger,
                            method=m,
                            dm=dm,
                            eps=privacy_budget,
                            folds=folds,
                            depth=depth,
                            ds=ds,
                            ds_struct=ds_struc,
                            ds_dom=ds_dom,
                            ds_name=d[0],
                            eps_total=e,
                        )


if __name__ == "__main__":
    set_seeds(10)
    experiment_dpid3()
