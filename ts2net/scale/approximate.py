"""
Approximate nearest-neighbor helpers for large panels (horizon 0.6).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from .._validation import validate_positive_int

_APPROX_THRESHOLD = 500


def has_pynndescent() -> bool:
    try:
        import pynndescent  # noqa: F401

        return True
    except ImportError:
        return False


def should_use_approximate(n_series: int, approximate: bool, threshold: int) -> bool:
    """Return whether approximate kNN is recommended."""
    if approximate:
        return True
    return n_series >= threshold and (has_pynndescent() or n_series < 10_000)


def approximate_knn_network(
    D: NDArray[np.float64],
    k: int = 5,
    weighted: bool = True,
    directed: bool = False,
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """
    Build a k-NN graph from a precomputed distance matrix.

    Uses scikit-learn ``NearestNeighbors`` with ``metric='precomputed'``.
    For very large panels in raw feature space, prefer
    ``net_knn_approx(X, metric='euclidean')`` (pynndescent).

    Parameters
    ----------
    D : array (n, n)
        Pairwise distance matrix.
    k : int, default 5
        Neighbors per node.
    weighted : bool, default True
        Store distances as edge weights.
    directed : bool, default False
        If True, return a directed graph.

    Returns
    -------
    G : networkx.Graph or DiGraph
    A : adjacency matrix
    """
    k = validate_positive_int("k", k)
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k must be in range [1, {n - 1}], got {k}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="precomputed")
    nn.fit(D)
    distances, indices = nn.kneighbors(D)

    A = np.zeros((n, n), dtype=np.float64)
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j_idx in range(1, k + 1):
            j = int(indices[i, j_idx])
            dist = float(distances[i, j_idx])
            weight = dist if weighted else 1.0
            if directed:
                A[i, j] = weight
                if weighted:
                    graph.add_edge(i, j, weight=weight)
                else:
                    graph.add_edge(i, j)
            else:
                A[i, j] = A[j, i] = max(A[i, j], weight)
                if i < j:
                    if weighted:
                        graph.add_edge(i, j, weight=weight)
                    else:
                        graph.add_edge(i, j)

    return graph, A


def approximate_knn_panel(
    X: NDArray[np.float64],
    k: int = 5,
    metric: str = "euclidean",
    weighted: bool = True,
    directed: bool = False,
    n_neighbors: int | None = None,
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """
    Approximate k-NN on a feature panel using pynndescent (fast at large n).

    Requires ``pip install ts2net[approx]``.
    """
    from ..multivariate.builders import net_knn_approx

    k = validate_positive_int("k", k)
    if n_neighbors is None:
        n_neighbors = max(k, 15)
    return net_knn_approx(
        X,
        k=k,
        metric=metric,
        n_neighbors=n_neighbors,
        weighted=weighted,
        directed=directed,
    )
