from multiprocessing.pool import ThreadPool
from tree import (
    build_random_tree,
    set_majority,
    set_majority_dp,
    set_majority_pf,
    set_majority_rnm,
    set_majority_ldp,
    Tree,
    predict as tree_predict,
)
from collections import Counter
from typing import cast
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import copy

from utils.experiment_type import ExperimentType


def split_dataframe(df: pl.DataFrame, n_chunks: int):
    if df.shape[0] % (n_chunks) == 0:
        rows_per_chunks = df.shape[0] // (n_chunks)
    else:
        rows_per_chunks = df.shape[0] // (n_chunks - 1)

    return map(
        lambda d: d.select(pl.exclude("idx")),
        df.with_row_count("idx")
        .with_column(pl.col("idx").apply(lambda i: int(cast(int, i) / rows_per_chunks)))
        .partition_by("idx"),
    )


def task_build_tree(
    build_type: ExperimentType,
    c: pl.DataFrame,
    max_depth: int,
    continuous_feat: dict[str, tuple[int, int]],
    discrete_feat: dict[str, list[str]],
    budget: float,
    global_sens: float,
    label_col_name: str,
    classes: list[str],
) -> Tree:
    assert c.shape[0] > 0  # shoud have at least one row

    tree = build_random_tree(
        c, max_depth, copy.deepcopy(continuous_feat), copy.deepcopy(discrete_feat), label_col_name, classes
    )
    if build_type == ExperimentType.DEFAULT:
        set_majority(tree, label_col_name=label_col_name)
    elif build_type == ExperimentType.DP:
        set_majority_dp(tree, budget, 1.0, label_col_name=label_col_name) # TODO: global sens
    elif build_type == ExperimentType.PF:
        set_majority_pf(tree, budget, 1.0) #TODO: global_sens
    elif build_type == ExperimentType.RLNM_LAPLACE:
        set_majority_rnm(tree, budget, global_sens, label_col_name=label_col_name)
    elif build_type == ExperimentType.RLNM_EXPONENTIAL:
        set_majority_rnm(tree, budget, global_sens, label_col_name=label_col_name, dist="exponential")
    elif build_type == ExperimentType.LOCAL_DAMPENING:
        set_majority_ldp(tree, budget, label_col_name=label_col_name)
    else:
        raise ValueError("The build type should be set as valid value.")

    return tree


def build_forest(
    build_type: ExperimentType,
    data: pl.DataFrame,
    n_trees: int,
    max_depth: int,
    continuous_feat: dict[str, tuple[int, int]],
    discrete_feat: dict[str, list[str]],
    budget: float,
    global_sens: float,
    label_col_name: str,
    classes: list[str],
) -> list[Tree]:
    forest = []
    chunks = list(split_dataframe(data, n_trees))

    pool = ThreadPool(processes=1)
    jobs_set = []
    for c in chunks:
        jobs_set.append(
            pool.apply_async(
                task_build_tree,
                (
                    build_type,
                    copy.deepcopy(c),
                    max_depth,
                    continuous_feat,
                    discrete_feat,
                    budget,
                    global_sens,
                    label_col_name,
                    classes,
                ),
            )
        )

    pool.close()
    pool.join()

    for j in jobs_set:
        forest.append(j.get())

    return forest


def predict(forest: list[Tree], data: pl.DataFrame) -> pl.DataFrame:
    predictions = pl.DataFrame()
    for i, t in enumerate(forest):
        p = tree_predict(t, data)
        df_pred = pl.DataFrame({f"tree_{i}": p})
        predictions = pl.concat([predictions, df_pred], how="horizontal")

    return predictions.apply(lambda r: Counter(r).most_common(1)[0][0])
