from numpy import ndarray

class Selection:
    sens: float

    def __call__(self, u: ndarray) -> int:
        raise NotImplemented("The selection algorithm should be implemented.")
