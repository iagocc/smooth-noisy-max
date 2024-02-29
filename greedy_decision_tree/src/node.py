from __future__ import annotations
from dataclasses import dataclass, field
from operator import itemgetter
from collections import Counter
from typing import TypeVar, Generic

from src.attribute import AttrVal

T = TypeVar("T")


@dataclass
class Node(Generic[T]):
    attr: str | int | None
    split_point: T | None
    children: list[Node] = field(default_factory=lambda: [])


@dataclass(kw_only=True)
class Leaf(Node):
    children: None
    label_counts: Counter
    major: AttrVal | None

    def majority(self):
        return max(self.label_counts, key=itemgetter(1))
