from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Sequence, Callable, Optional, Tuple


def net_kout(
    D: np.ndarray,
    k: int,
    names: Optional[Sequence[str]] = None,
    weight_fn: Optional[Callable[[float], float]] = None,
) -> Tuple[nx.DiGraph, np.ndarray]:
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1 : k + 1]
    A = np.zeros((n, n), float)
    for i in range(n):
        for j in idx[i]:
            w = 1.0 if weight_fn is None else float(weight_fn(float(D[i, j])))
            A[i, j] = w
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    if names is not None:
        G = nx.relabel_nodes(G, {i: names[i] for i in range(n)})
    return G, A
