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
            np.array([self.demand[v] * 1000 for k, v in sorted(self.look_up.items(), key=lambda item: item[1])]),
            np.array(list(min(v, 100) for v in self.edges.values())),
            root,
            n,
            algo,
            loglevel
        )
