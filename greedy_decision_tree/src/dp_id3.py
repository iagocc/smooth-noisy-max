from os import error
from uu import Error
from src.dispersion.info_gain import InfoGainFriedman
from src.tree import Tree
from src.node import Leaf, Node
from src.attribute import AttrVal

from dataclasses import dataclass, field
from collections import Counter
from random import choices
from typing import cast

import numpy as np


@dataclass
class DpID3(Tree):
    dm: InfoGainFriedman
    dataset: np.ndarray
    dataset_struct: np.ndarray
    dataset_domain: dict[int, list]
    max_depth: int
    min_sample: int
    eps: float
    _C: list = field(init=False)

    def __post_init__(self):
        self._C = list(set(self.dataset[:, -1]))

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        predictions = []
        for r in dataset:
            node = self.root

            if not node:
                raise Error("There is no root.")

            while node.children:
                for c in node.children:
                    if r[c.attr] == c.split_point:
                        node = c
                        break

            leaf: Leaf = cast(Leaf, node)
            predictions.append(leaf.major)

        return np.array(predictions)

    def fit(self, dataset: np.ndarray) -> None:
        node = Node(None, None)
        self.root = self._build_tree(node, dataset, set(np.arange(self.dataset_struct.shape[0])), self.max_depth)

    def _get_domain(self, dataset: np.ndarray, attr: int) -> np.ndarray:
        return np.array(list(set(dataset[:, attr])))

    def _build_tree(self, node: Node, dataset: np.ndarray, attr_set: set[int], depth: int) -> Node:
        _, m = dataset.shape
        t = max([len(list(set(dataset[:, col]))) for col in range(m)]) / 5
        Nt = self._noisy_count(dataset.shape[0])

        if (t == 0) or (m == 0) or (len(attr_set) == 0) or (depth == 0) or (Nt / (t * len(self._C)) <= np.sqrt(2) / self.eps):
            Tc = [self._partition(dataset, col_idx=-1, val=c) for c in self._C]
            Nc = [self._noisy_count(t.shape[0]) for t in Tc]
            return Leaf(attr=node.attr, split_point=node.split_point, children=None, label_counts=Counter(dataset[:, -1]), major=self._C[np.argmax(Nc)])

        igs = np.array([self.dm(data=dataset, attr=col) for col in attr_set])
        selected_idx = self.selection(igs)
        selected = list(attr_set)[selected_idx]

        attr_set.remove(selected)

        for v in self.dataset_domain[selected]:
            node_v = Node(selected, v)
            T_i = self._partition(dataset=dataset, col_idx=selected, val=v)
            np.delete(dataset, selected)
            node.children.append(self._build_tree(node_v, T_i, attr_set, depth - 1))

        return node

    def _noisy_count(self, count: int) -> float:
        # Without DP
        # if self.selection.__class__.__name__ == "DefaultSelection":
        #     return count
        return count + np.random.default_rng().laplace(loc=0, scale=(1 / self.eps), size=1)[0]

    def _partition(self, dataset: np.ndarray, col_idx: int, val: AttrVal) -> np.ndarray:
        mask = dataset[:, col_idx] == val
        return dataset[mask, :]
