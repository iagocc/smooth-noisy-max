from dataclasses import dataclass

import numpy as np

from src.selection.selection import Selection


@dataclass
class RNM(Selection):
    eps: float
    sens: float

    def __call__(self, u: np.ndarray) -> int:
        noise = np.random.default_rng().exponential(scale=((2 * self.sens) / self.eps), size=u.size)
        return int(np.argmax(u + noise))


class RLNM(RNM):
    def __call__(self, u: np.ndarray) -> int:
        return super().__call__(u)

@dataclass
class TStudentRLNM(RNM):
    beta: float

    @staticmethod
    def get_beta(eps, d):
        return eps/(2*(d+1))
    
    def __call__(self, u: np.ndarray) -> int:
        def rv(d: int):
            X = np.random.normal(size=d+1)
            return X[0] / np.sqrt((sum(X[1:]**2))/d)

        d = 3
        beta = self.beta
        Z = np.array([rv(3) for i in range(u.size)])

        s = 2 * np.sqrt(d) * (self.eps - abs(beta) * (d+1)) / (d+1)
        noise = ((2*self.sens)/s)*Z

        return int(np.argmax(u + noise))

@dataclass
class LlnRLNM(RNM):
    beta: float

    @staticmethod
    def get_beta(eps):
        return eps/2

    def __call__(self, u: np.ndarray) -> int:
        def rv(sigma: float, size: int):
            X = np.random.laplace(size=size)
            Y = np.random.normal(size=size)
            Z = X * np.exp(sigma * Y)
            return Z

        beta = self.beta
        opt_sigma = np.real(np.roots([5 * self.eps / beta, -5, 0, -1])[0])
        Z = rv(opt_sigma, u.size)
        alpha = np.exp(-(3/2)*(opt_sigma**2)) * (self.eps - (abs(beta)/abs(opt_sigma)))
        noise = ((2*self.sens)/alpha)*Z

        return int(np.argmax(u + noise))

@dataclass
class SmoothLaplaceRLNM(RNM):
    delta: float

    @staticmethod
    def get_beta(eps, delta):
        return eps/(2*np.log(2/delta))
    
    def __call__(self, u: np.ndarray) -> int:
        noise = np.random.default_rng().laplace(loc=0, scale=(2*self.sens)/self.eps, size=u.size)
        return int(np.argmax(u + noise))
