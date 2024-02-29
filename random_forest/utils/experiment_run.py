import numpy as np

from tree import Tree
from dataclasses import dataclass


@dataclass
class ExperimentRun:
    dataset: str
    label_col_name: str
    execution_times: int
    n_trees: int
    max_depth: int
    budgets: list[float]


def generate_metrics(
    n_trees: int,
    max_depth: int,
    eps: float,
    forest: list[Tree],
    y_test: np.ndarray,
    predicted: np.ndarray,
    accuracy: float,
    f1: float,
):
    metric = {}
    metric["y"] = y_test.tolist()
    metric["y_pred"] = predicted.tolist()
    metric["accuracy"] = accuracy
    metric["f1"] = f1
    metric["n_trees"] = n_trees
    metric["budget"] = eps
    metric["max_depth"] = max_depth
    metric["amount_leaves"] = sum([len(t.leaves) if t.leaves else 0 for t in forest])
    metric["amount_empty_leaves"] = sum([t.amount_empty_leaves for t in forest])
    metric["proportion_empty_leaves"] = metric["amount_empty_leaves"] / metric["amount_leaves"]
    return metric
