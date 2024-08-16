"""
Finally the time has come to solve something...
"""
from abc import ABC, abstractmethod

import numpy as np

from pcst_fast import pcst_fast


class AbstractSolver(ABC):
    def __init__(self, edges, look_up, demand):
        self.edges = edges
        self.look_up = look_up
        self.demand = demand
        super().__init__()

    @abstractmethod
    def solve(self):
        pass


class PCSTSolver(AbstractSolver):
    def solve(self, root=-1, algo='strong', loglevel=0, n=1):
        self.s_vertices, self.s_edges = pcst_fast(
            np.array(list(self.edges.keys())),
            np.array([10_000_000 if self.demand[i] > 0 else 0 for i in range(len(self.demand))]),
            np.array(list(v for v in self.edges.values())),
            root,
            n,
            algo,
            loglevel
        )
