"""
Similarity networks from distance matrices.

Wraps ``ts_dist`` and network builders into a single high-level API.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int
from ..multivariate.builders import net_enn, net_knn, net_weighted
from ..multivariate.distances import ts_dist
from ..scale.approximate import (
    approximate_knn_network,
    approximate_knn_panel,
    has_pynndescent,
    should_use_approximate,
)
from .distances_extra import (
    matrix_profile_distance_matrix,
    soft_dtw_distance_matrix,
)

SimilarityMethod = Literal[
    "euclidean",
    "correlation",
    "spearman",
    "dtw",
    "soft_dtw",
    "matrix_profile",
    "ccf",
    "nmi",
]
NetworkRule = Literal["knn", "epsilon", "threshold", "complete"]


def _euclidean_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy.spatial.distance import pdist, squareform

    return squareform(pdist(X, metric="euclidean"))


def _spearman_distance_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy import stats

    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = stats.spearmanr(X[i], X[j])
            d = 1.0 - abs(rho) if np.isfinite(rho) else 1.0
            D[i, j] = D[j, i] = d
    return D


def similarity_matrix(
    X: NDArray[np.float64],
    method: SimilarityMethod = "correlation",
    n_jobs: int = 1,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Pairwise dissimilarity matrix between time series.

    Parameters
    ----------
    X : array (n_series, n_points)
    method : str
        ``euclidean``, ``correlation``, ``spearman``, or any ``ts_dist`` method.
    n_jobs : int
        Parallel workers for supported methods.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if method == "euclidean":
        return _euclidean_matrix(X)
    if method == "spearman":
        return _spearman_distance_matrix(X)
    if method == "soft_dtw":
        gamma = float(kwargs.pop("gamma", 1.0))
        return soft_dtw_distance_matrix(X, gamma=gamma)
    if method == "matrix_profile":
        subseq_len = int(kwargs.pop("subseq_len", 10))
        return matrix_profile_distance_matrix(X, subseq_len=subseq_len)

    return ts_dist(X, method=method, n_jobs=n_jobs, **kwargs)


def similarity_network(
    X: NDArray[np.float64],
    method: SimilarityMethod = "correlation",
    rule: NetworkRule = "knn",
    k: int = 5,
    epsilon: float = 0.3,
    threshold: float | None = None,
    n_jobs: int = 1,
    weighted: bool = True,
    approximate: bool = False,
    approx_threshold: int = 500,
    **kwargs,
) -> tuple[nx.Graph, NDArray[np.float64]]:
    """
    Build a similarity network from a panel of time series.

    Parameters
    ----------
    X : array (n_series, n_points)
    method : str
        Distance/similarity measure.
    rule : {"knn", "epsilon", "threshold", "complete"}
    k, epsilon, threshold
        Sparsification parameters.
    n_jobs : int
        Parallel workers for distance computation.
    weighted : bool
        Edge weights = dissimilarity values.
    approximate : bool, default False
        Use pynndescent approximate k-NN when ``rule='knn'``.
    approx_threshold : int, default 500
        Auto-enable approximate k-NN when ``n_series >= approx_threshold``.

    Returns
    -------
    G : networkx.Graph
    D : distance matrix
    """
    D = similarity_matrix(X, method=method, n_jobs=n_jobs, **kwargs)
    n_series = X.shape[0]

    if rule == "knn":
        k_val = validate_positive_int("k", k)
        if should_use_approximate(n_series, approximate, approx_threshold):
            if method == "euclidean" and has_pynndescent():
                G, _ = approximate_knn_panel(
                    X, k=k_val, metric="euclidean", weighted=weighted
                )
            else:
                G, _ = approximate_knn_network(D, k=k_val, weighted=weighted)
        else:
            G, _ = net_knn(D, k=k_val, weighted=weighted)
    elif rule == "epsilon":
        G, _ = net_enn(D, eps=epsilon, weighted=weighted)
    elif rule == "threshold":
        if threshold is None:
            threshold = np.percentile(D[D > 0], 20) if np.any(D > 0) else 0.5
        D_thr = D.copy()
        D_thr[D_thr > threshold] = 0.0
        G, _ = net_weighted(D_thr, directed=False)
    elif rule == "complete":
        G, _ = net_weighted(D, directed=False)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return G, D
