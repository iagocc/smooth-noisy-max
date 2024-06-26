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

class TStudentRLNM(RNM):
    def __call__(self, u: np.ndarray) -> int:
        def rv(d: int):
            X = np.random.normal(size=d+1)
            return X[0] / np.sqrt((sum(X[1:]**2))/d)

        d = 3
        beta = self.eps/(2*(d+1))
        Z = np.array([rv(3) for i in range(u.size)])

        s = 2 * np.sqrt(d) * (self.eps - abs(beta) * (d+1)) / (d+1)
        noise = ((2*self.sens)/s)*Z

        return int(np.argmax(u + noise))
    
class LlnRLNM(RNM):
    def __call__(self, u: np.ndarray) -> int:
        def rv(sigma):
            X = np.random.laplace()
            Y = np.random.normal()
            Z = X * np.exp(sigma * Y)
            return Z

        beta = self.eps/2
        opt_sigma = np.real(np.roots([5 * self.eps / beta, -5, 0, -1])[0])
        Z = rv(opt_sigma)
        alpha = np.exp(-(3/2)*(opt_sigma**2)) * (self.eps - (abs(beta)/abs(opt_sigma)))
        noise = ((2*self.sens)/alpha)*Z

        return int(np.argmax(u + noise))
