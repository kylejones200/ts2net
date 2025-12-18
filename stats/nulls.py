"""
nulls.py - Null model generators for network analysis.

This module provides functions for generating null models of networks, which are useful
for statistical testing and hypothesis generation in network analysis. It includes
implementations of Erdős–Rényi random graphs and configuration models.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Iterable, Optional


def er_from_names(
    names: Iterable[str], p: float, seed: Optional[int] = 3363
) -> nx.Graph:
    rng = np.random.default_rng(seed)
    names = list(names)
    n = len(names)
    G = nx.Graph()
    G.add_nodes_from(names)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() <= p:
                G.add_edge(names[i], names[j])
    return G


def config_shuffle(
    G: nx.Graph,
    nswap: int | None = None,
    max_tries: int | None = None,
    seed: Optional[int] = 3363,
) -> nx.Graph:
    H = G.copy()
    m = H.number_of_edges()
    if nswap is None:
        nswap = 5 * m
    if max_tries is None:
        max_tries = 100 * m
    rng = np.random.default_rng(seed)
    try:
        nx.double_edge_swap(
            H, nswap=nswap, max_tries=max_tries, seed=int(rng.integers(1, 2**31 - 1))
        )
    except Exception:
        pass
    return H


def er_like(G: nx.Graph, seed: Optional[int] = 3363) -> nx.Graph:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 0.0 if n < 2 else 2.0 * m / (n * (n - 1))
    return er_from_names(G.nodes(), p, seed=seed)
