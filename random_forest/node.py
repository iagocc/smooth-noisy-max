from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, TypeVar, Generic, Any
from polars import DataFrame

import uuid

T = TypeVar("T")


@dataclass
class Domain:
    bottom: float
    top: float


@dataclass
class Node(Generic[T]):
    attribute: str | None
    value: T
    parent: Node | None
    histogram: DataFrame
    children: List[Node] = field(default_factory=list)
    majority: Any = field(default=None)
    level: int = field(default=-1)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.node_id[:6] + "#" + str(self.level)

    def __repr__(self) -> str:
        return self.node_id[:6] + "#" + str(self.level)
