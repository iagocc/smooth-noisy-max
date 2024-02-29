from operator import itemgetter
from collections import Counter
from numpy import ndarray
from dataclasses import dataclass

from src.attribute import AttrType
from src.node import Node, Leaf
from src.tree import Tree
from src.selection.selection import Selection

import numpy as np


@dataclass
class MultinomialTree(Tree):
    dataset_struct: np.ndarray
    structure_ratio: float
    max_depth: int
    eps: float
    min_leaf_size: int
    thr_selection: Selection
    maj_selection: Selection
    normalize: bool = False

    def predict(self, dataset: ndarray) -> ndarray:
        predictions = []
        for i, r in enumerate(dataset):
            node = self.root
            while node.children and len(node.children) > 0:
                left = node.children[0]
                right = node.children[1]
                if node.split_point and (
                    (self.dataset_struct[node.attr] == AttrType.C.value and r[node.attr] >= node.split_point)
                    or (self.dataset_struct[node.attr] == AttrType.D.value and r[node.attr] == node.split_point)
                ):
                    node = right
                else:
                    node = left
            leaf: Leaf = node
            predictions.append(leaf.major)

        return np.array(predictions)

    def fit(self, dataset: ndarray) -> None:
        self.root = self._build_tree(dataset, self.dataset_struct, 0)

    def _build_tree(self, dataset: ndarray, attr_struct: ndarray, depth: int) -> Node:
        size_structure: int = int(np.floor(dataset.shape[0] * self.structure_ratio))
        ds, de = dataset[:size_structure, :], dataset[size_structure:, :]

        # (attr, split_point, info_gain)
        V = {}
        I = {}

        if (de.shape[0] > self.min_leaf_size) and (depth <= self.max_depth):
            for attr, typ in enumerate(attr_struct):
                vals = list(set(ds[:, attr]))

                vij = [(attr, t, self.dm(ds, self._split_dataset(ds, attr, typ, t))) for t in self._get_split_points(vals)]

                V[attr] = vij
                I[attr] = max(vij, key=itemgetter(2))

            selected_attr = self._selection([i[2] for i in list(I.values())], "att")  # position 2 = info gain

            split_point_idx = self._selection([ij[2] for ij in V[selected_attr]], "thr")
            split_point = V[selected_attr][split_point_idx][1]

            r_data_idx = self._split_dataset(dataset, attr, attr_struct[attr], split_point)

            node = Node(attr, split_point)

            left_data = dataset[~r_data_idx]
            if left_data.shape[0] <= self.min_leaf_size:
                left = self._create_leaf(de)
            else:
                left = self._build_tree(left_data, attr_struct, depth + 1)
            node.children.append(left)

            right_data = dataset[r_data_idx]
            if right_data.shape[0] <= self.min_leaf_size:
                right = self._create_leaf(de)
            else:
                right = self._build_tree(right_data, attr_struct, depth + 1)
            node.children.append(right)

            return node
        else:
            return self._create_leaf(de)

    def _create_leaf(self, de):
        counter = Counter(de[:, -1])
        major_idx = self._selection(np.array(list(counter.values())), step="maj")
        major = list(counter.keys())[major_idx]
        return Leaf(attr=None, split_point=None, children=None, label_counts=counter, major=major)

    def _selection(self, I: ndarray, step: str) -> int:
        if self.normalize:
            I = self._minmax_softmax(I)

        if step == "maj":
            return self.maj_selection(np.array(I))
        elif step == "thr":
            return self.thr_selection(np.array(I))
        elif step == "att":
            return self.selection(np.array(I))
        else:
            raise ValueError("The step should be valid.")

    def _minmax_softmax(self, I: list) -> ndarray:
        min_i, max_i = min(I), max(I)
        if min_i == max_i:
            [1 / len(I) for _ in range(len(I))]
        else:
            I = [(i - min_i) / (max_i - min_i) for i in I]  # Î
        I = np.exp(I) / np.sum(np.exp(I), axis=0)  # softmax
        return I

    def _get_split_points(self, vals: set) -> ndarray:
        split_points = list(vals)
        if type == AttrType.C.value:
            split_points = np.convolve(vals, np.ones(2) / 2, mode="valid")  # moving average with size=2
        return split_points

    def _split_dataset(self, dataset, attr, type, t) -> ndarray:
        if type == AttrType.C.value:
            return dataset[:, attr] >= t
        elif type == AttrType.D.value:
            return dataset[:, attr] == t
