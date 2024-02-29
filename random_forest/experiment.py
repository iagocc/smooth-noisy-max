from pathlib import Path
import time
import pandas as pd
import copy
import pickle

import numpy as np
import polars as pl
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from rich.progress import Progress

from forest import build_forest, predict as forest_predict, ExperimentType

from utils.seed import set_all_seeds
from utils.experiment_run import ExperimentRun, generate_metrics
from utils.experiment_logger import ExperimentLogger
from utils.dataset import (
    clean_dataframe_column_names,
    extract_cont_disc_features,
    features_2dict,
    generate_dataset_summary,
)


def exec(
    name: str,
    type: ExperimentType,
    times: int,
    n_trees: int,
    max_depth: int,
    df: pl.DataFrame,
    label_col_name: str,
    classes: list[str],
    continuous: dict[str, tuple[int, int]],
    discrete: dict[str, list[str]],
    le: LabelEncoder,
    eps: float = 0.0,
    sens: float = 0.0,
):
    metrics = []
    for i in range(times):
        logger = ExperimentLogger()

        sample_rows = random.sample(range(0, df.shape[0]), int(df.shape[0] * 0.8))

        train = df.filter(pl.col("id").is_in(sample_rows))
        test = df.filter(~pl.col("id").is_in(sample_rows))

        forest = build_forest(
            type,
            train.select(pl.exclude(["id"])),
            n_trees,
            max_depth,
            copy.deepcopy(continuous),
            copy.deepcopy(discrete),
            eps,
            sens,
            label_col_name,
            classes,
        )

        y_test = test.select(pl.col(label_col_name)).to_numpy().ravel()
        predicted = forest_predict(forest, test.select(pl.exclude(["id"]))).to_numpy().ravel()
        accuracy = accuracy_score(le.transform(y_test), le.transform(predicted))
        f1 = f1_score(le.transform(y_test), le.transform(predicted), average="macro")

        metric = generate_metrics(n_trees, max_depth, eps, forest, y_test, predicted, float(accuracy), float(f1))

        metrics.append(logger.gen(metric))

    result_df = pd.DataFrame(metrics)
    if not eps:  # to fancy our filename
        eps = 0
    result_df.to_csv(f"result/{name}/logs/metrics_{type.value}_{eps}.csv", index=False)


def execute_experiment(
    name: str,
    label_col_name: str,
    times: int,
    n_trees: int,
    max_depth: int,
    budgets: list[float],
    opt_out: list[ExperimentType] | None = None,
):
    set_all_seeds(42)

    le = LabelEncoder()

    if Path(f"data/{name}.parquet").is_file():
        data = pd.read_csv(f"data/{name}.parquet")
    else:
        data = pd.read_csv(f"data/{name}.csv")

    clean_dataframe_column_names(data)
    y = data.pop(label_col_name)

    le.fit(y)

    continuos_feat, discrete_feat = extract_cont_disc_features(data)
    continuous, discrete = features_2dict(data, continuos_feat, discrete_feat)

    data[label_col_name] = y.astype(str)
    del y

    generate_dataset_summary(
        name,
        data,
        list(continuous.keys()),
        list(discrete.keys()),
        data[label_col_name].unique().tolist(),
        label_col_name,
    )

    df = pl.from_pandas(data).with_row_count("id")

    n_largest = (
        df.groupby(label_col_name).count().select(pl.col("count").top_k(2))
    )  # needled for the smoothed version proposed by Sam Fletcher et. al.
    j = n_largest[0, "count"] - n_largest[1, "count"]

    # the stardard method will run every time
    if opt_out is None:
        opt_out = [ExperimentType.DEFAULT]
    elif ExperimentType.DEFAULT not in opt_out:
        opt_out.append(ExperimentType.DEFAULT)

    run_experiment_types = set(ExperimentType) - set(opt_out)

    with Progress() as progress:
        task_eps = progress.add_task(
            f"Running the experiment! [{name}]", total=len(budgets) * len(run_experiment_types)
        )
        classes = data[label_col_name].unique().tolist()

        exec(
            name,
            ExperimentType.DEFAULT,
            times,
            n_trees,
            max_depth,
            df,
            label_col_name,
            classes,
            continuous,
            discrete,
            le,
        )

        for et in run_experiment_types:
            for eps in budgets:
                progress.update(task_eps, advance=1, description=f"{et.value} - {name} - ε = {eps}")

                smooth_local_sens = np.exp(np.clip(-1 * j * eps, -709.78, 709.78))  # clip to avoid overflow
                sensitivity = 1 if (et == ExperimentType.DP) or (et == ExperimentType.PF) else smooth_local_sens # in the local dampening setting the sensitivity is useless in out code

                exec(
                    name,
                    et,
                    times,
                    n_trees,
                    max_depth,
                    df,
                    label_col_name,
                    classes,
                    continuous,
                    discrete,
                    le,
                    eps,
                    sensitivity,
                )


def execute_battery(settings: list[ExperimentRun], opt_out: list[ExperimentType] | None = None):
    start_time = time.time()
    for e in settings:
        e_start_time = time.time()
        execute_experiment(e.dataset, e.label_col_name, e.execution_times, e.n_trees, e.max_depth, e.budgets, opt_out)
        print(f" --- {e.dataset} done in {time.time() - e_start_time} seconds --- ")

    print(f"|--- battery done in {time.time() - start_time} seconds ---|")


budgets = [0.01, 0.05, 0.1, 1, 2]
ntrees = 32
times = 10

battery = [
    # ExperimentRun("adult", "income", times, ntrees, 9, budgets),
    # ExperimentRun("mushroom", "class_name", times, ntrees, 11, budgets),
    ExperimentRun("nursery", "final_evaluation", times, ntrees, 4, budgets),
    # ExperimentRun("gamma", "class_name", times, ntrees, 8, budgets),
    # ExperimentRun("pendigits", "class_name", times, ntrees, 12, budgets),
    # ExperimentRun("wallsensor", "class_name", times, ntrees, 4, budgets),
    # ExperimentRun("compass", "decile_score", times, ntrees, 5, budgets),
    # ExperimentRun("wine", "quality", times, ntrees, 10, budgets),
]
out = list(set(ExperimentType) - set([ExperimentType.PF]))
# out = [ExperimentType.DEFAULT, ExperimentType.SMOOTHED_DP]
execute_battery(battery, opt_out=out)
