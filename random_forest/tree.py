from __future__ import annotations
from dataclasses import dataclass, field
from random import randrange, choices
from typing import List

import polars as pl
import numpy as np
import math
import mpmath as mp

from node import Node, Domain
from dampening import dampening_function

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Tree:
    root: Node
    leaves: List[Node] | None = field(default=None)
    amount_empty_leaves: int = field(default=0)


def build_random_tree(
    df: pl.DataFrame,
    max_depth: int,
    continuous_feat: dict[str, tuple[int, int]],
    discrete_feat: dict[str, List[str]],
    label_col_name: str,
    classes: list[str],
) -> Tree:
    def build_node(parent: Node, current_depth: int) -> Node:
        if current_depth <= max_depth and (len(continuous_feat) + len(discrete_feat) > 0):
            num_features = len(continuous_feat) + len(discrete_feat)
            selected_idx = randrange(0, num_features)
            empty_hist = pl.DataFrame({label_col_name: classes, "count": np.zeros(len(classes))})

            if selected_idx < len(continuous_feat):  # random was chosen one continuos feature
                attr_name = list(continuous_feat.keys())[selected_idx]

                min_domain = continuous_feat[attr_name][0]
                max_domain = continuous_feat[attr_name][1]
                split_value = np.random.uniform(low=min_domain, high=max_domain, size=1).astype(int)[0]

                node = Node[Domain](
                    attribute=attr_name,
                    value=Domain(float("-inf"), split_value),
                    parent=parent,
                    level=current_depth,
                    histogram=empty_hist,
                )
                parent.children.append(build_node(node, current_depth=current_depth + 1))

                node = Node[Domain](
                    attribute=attr_name,
                    value=Domain(split_value, float("inf")),
                    parent=parent,
                    level=current_depth,
                    histogram=empty_hist,
                )
                parent.children.append(build_node(node, current_depth=current_depth + 1))

            else:  # random was chosen one discrete feature
                attr_name = list(discrete_feat.keys())[selected_idx - len(continuous_feat)]
                values = discrete_feat.pop(attr_name)

                for v in values:
                    node = Node[str](
                        attribute=attr_name,
                        value=v,
                        parent=parent,
                        level=current_depth,
                        histogram=empty_hist,
                    )
                    parent.children.append(build_node(node, current_depth=current_depth + 1))

        return parent

    tree = Tree(root=build_node(Node[int](None, 0, None, level=0, histogram=pl.DataFrame()), 1))
    tree = train(tree, df, label_col_name)
    return tree


def train(tree: Tree, data: pl.DataFrame, label_col_name: str) -> Tree:
    for row in data.iter_rows(named=True):
        node = traverse(tree, row._asdict())
        label = getattr(row, label_col_name)
        hist = node.histogram.with_columns(
            pl.when(pl.col(label_col_name) == label).then(pl.col("count") + 1).otherwise(pl.col("count"))
        )
        node.histogram = hist

    return tree


def traverse(tree: Tree, row: dict):
    node = tree.root
    while node.children:
        for child in node.children:
            if type(child.value) == Domain:
                if (row[child.attribute] <= child.value.top) and (row[child.attribute] >= child.value.bottom):
                    node = child
            else:
                if row[child.attribute] == child.value:  # if the node is categorical
                    node = child

    return node


def get_amount_empty_leaves(leaves: list[Node]) -> int:
    empties = 0
    for l in leaves:
        if l.histogram.select(pl.sum("count").alias("sum"))[0, 0]:
            empties += 1

    return empties


def get_all_leaves(tree: Tree) -> List[Node]:
    if tree.leaves:
        return tree.leaves

    root = tree.root
    leaves: list[Node] = []
    stack = [root]
    while stack:
        n = stack.pop()
        for i in n.children:
            stack.append(i)

        if not n.children or len(n.children) == 0:  # leaf is a node without child
            leaves.append(n)

    tree.leaves = leaves
    tree.amount_empty_leaves = get_amount_empty_leaves(leaves)
    return leaves


def set_majority(tree: Tree, label_col_name: str) -> None:
    leaves = get_all_leaves(tree)
    for l in leaves:
        l.majority = l.histogram.sort("count", reverse=True).select(pl.first(label_col_name))[
            0, 0
        ]  # set as major the value of highest count]

def set_majority_pf(tree: Tree, budget: float, sensitivity: float) -> None:
    leaves = get_all_leaves(tree)
    for l in leaves:
        vals = l.histogram.to_numpy()
        q = vals[:,1].astype(np.float_)
        q = q - q.max()
        p = np.exp((q*budget)/(2*sensitivity))

        for i in np.random.permutation(p.size):
            if np.random.rand() <= p[i]: # Bernoulli flips head
                l.majority = vals[i, 0]
                break



def set_majority_dp(tree: Tree, budget: float, sensitivity: float, label_col_name: str) -> None:
    leaves = get_all_leaves(tree)
    for l in leaves:
        max_idx = l.histogram.sort("count", reverse=True).select(pl.first(label_col_name))[0, 0]
        scores = {l[0]: 0 for l in l.histogram.select(pl.col(label_col_name)).unique().iter_rows()}
        scores[max_idx] = 1  # the majority score is one

        assert math.isclose(sum(scores.values()), 1.0)  # there is only one max

        # When necessary we use mpmath for handling with very small or big numbers
        if sensitivity >= 0.5:
            mech_scores = np.clip(
                (budget * np.array(list(scores.values()), dtype=np.float64)) / (2 * sensitivity), -709.78, 709.78
            )  # clip to avoid overflow
            exp_scores = np.array([float(np.exp(i)) for i in mech_scores], dtype=np.float64)
            sum_scores = np.sum(exp_scores)
            final_scores = (exp_scores / sum_scores).tolist()
        else:
            assert mp.exp is not None  # remove typing issue from mpmath library
            mech_scores = [mp.exp((budget * s) / (2 * sensitivity)) for s in scores.values()]
            sum_scores = mp.fsum(mech_scores)
            final_scores = [float(mp.fdiv(s, sum_scores)) for s in mech_scores]

        assert math.isclose(sum(final_scores), 1.0)

        major = choices(range(l.histogram.shape[0]), final_scores, k=1)[0]
        l.majority = list(scores.keys())[
            major
        ]  # python 3.7+ the dict is ordered https://mail.python.org/pipermail/python-dev/2017-December/151283.html

def t_beta(eps: float, d: int):
    return eps/(2*(d+1))

def t_noise(eps: float, sens: float, size: int):
    def rv(d: int):
        X = np.random.normal(size=d+1)
        return X[0] / np.sqrt((sum(X[1:]**2))/d)

    d = 3
    beta = t_beta(eps, d)
    Z = np.array([rv(3) for i in range(size)])

    s = 2 * np.sqrt(d) * (eps - abs(beta) * (d+1)) / (d+1)
    noise = ((2*sens)/s)*Z
    return noise

def lln_noise(eps: float, sens: float, size: int):
    def rv(sigma: float, size: int):
        X = np.random.laplace(size=size)
        Y = np.random.normal(size=size)
        Z = X * np.exp(sigma * Y)
        return Z

    beta = eps/2
    opt_sigma = np.real(np.roots([5 * eps / beta, -5, 0, -1])[0])
    Z = rv(opt_sigma, size)
    alpha = np.exp(-(3/2)*(opt_sigma**2)) * (eps - (abs(beta)/abs(opt_sigma)))
    noise = ((2*sens)/alpha)*Z
    return noise

def smooth_lap_beta(eps: float, delta=1e-6):
    return eps/(2*np.log(2/delta))

def smooth_lap(eps: float, sens: float, size: int):
    noise = np.random.default_rng().laplace(loc=0, scale=(2*sens)/eps, size=size)
    return noise

def set_majority_rnm(tree: Tree, budget: float, sens: float, label_col_name: str, dist: str = "laplace") -> None:
    leaves = get_all_leaves(tree)
    for l in leaves:
        max_idx = l.histogram.sort("count", reverse=True).select(pl.first(label_col_name))[0, 0]
        scores = {l[0]: 0 for l in l.histogram.select(pl.col(label_col_name)).unique().iter_rows()}
        scores[max_idx] = 1  # the majority score is one

        assert sum(scores.values()) == 1
        u = np.array(list(scores.values()))

        scale = (2 * sens) / budget
        if dist == "laplace":
            noise = np.random.default_rng().laplace(loc=0, scale=scale, size=l.histogram.shape[0])
        elif dist == "exponential":
            noise = np.random.default_rng().exponential(scale=scale, size=l.histogram.shape[0])
        elif dist == "t":
            noise = t_noise(budget, sens, u.size)
        elif dist ==  "lln":
            noise = lln_noise(budget, sens, u.size)
        elif dist == "smooth_lap":
            noise = smooth_lap(budget, sens, u.size)
        else:
            raise Exception("The distribution should be set as expected value (laplace or exponential)")

        scores_noisy = u + noise
        reported = np.argmax(scores_noisy)
        l.majority = list(scores.keys())[reported]


def set_majority_ldp(tree: Tree, budget: float, label_col_name: str) -> None:
    leaves = get_all_leaves(tree)
    for l in leaves:
        class_counts = {l[0]: l[1] for l in l.histogram.select(pl.col([label_col_name, "count"])).unique().iter_rows()}
        counts = np.array(list(class_counts.values()))
        damped_scores = [dampening_function(counts, ci) for ci in range(len(class_counts.keys()))]

        assert mp.exp is not None  # remove typing issue from mpmath library
        mech_scores = [mp.exp((budget * s) / (2)) for s in damped_scores]
        sum_scores = mp.fsum(mech_scores)
        final_scores = [float(mp.fdiv(s, sum_scores)) for s in mech_scores]

        assert math.isclose(sum(final_scores), 1.0)

        major = choices(range(l.histogram.shape[0]), final_scores, k=1)[0]
        l.majority = list(class_counts.keys())[
            major
        ]  # python 3.7+ the dict is ordered https://mail.python.org/pipermail/python-dev/2017-December/151283.html


def predict(tree: Tree, data: pl.DataFrame) -> list:
    def predict_row(row: dict) -> str:
        node = traverse(tree, row)
        if not node.majority:
            raise Exception("The row prediction will return None.")

        return node.majority

    predicted = [predict_row(row._asdict()) for row in data.iter_rows(named=True)]
    return predicted


def print_tree(tree: Tree):
    root = tree.root
    stack = [root]
    while stack:
        n = stack.pop()
        for i in n.children:
            stack.append(i)

        print(" " * n.level + f"{n.attribute} v: {n.value}")
