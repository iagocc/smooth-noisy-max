
from src.node import Node
from src.dispersion.dispersion import DispersionMeasure
from src.selection.selection import Selection

from numpy import ndarray

from dataclasses import dataclass

@dataclass
class Tree():
    root: Node | None
    dm: DispersionMeasure
    selection: Selection

    def fit(self, dataset: ndarray, attr_struct: ndarray) -> None:
        pass

    def predict(self, dataset: ndarray) -> ndarray:
        pass