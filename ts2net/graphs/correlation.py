"""
Correlation-based networks from multivariate time series panels.

Supports Pearson, Spearman, Kendall, and distance correlation with
thresholded, k-NN, or epsilon network construction.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .._validation import validate_positive_int
from ..multivariate.builders import net_enn, net_knn, net_weighted
from ..stats.stats import partial_corr

CorrelationMethod = Literal["pearson", "spearman", "kendall", "dcor"]
NetworkRule = Literal["knn", "epsilon", "threshold", "complete"]


def _pairwise_correlation(
    x: NDArray[np.float64], y: NDArray[np.float64], method: str
) -> float:
    if method == "pearson":
        rho = np.corrcoef(x, y)[0, 1]
        return float(rho) if np.isfinite(rho) else 0.0
    if method == "spearman":
        rho, _ = stats.spearmanr(x, y)
        return float(rho) if np.isfinite(rho) else 0.0
    if method == "kendall":
        rho, _ = stats.kendalltau(x, y)
        return float(rho) if np.isfinite(rho) else 0.0
    if method == "dcor":
        return _distance_correlation(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


def _distance_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Distance correlation in [0, 1]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("Series must have same length")

    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a_mean_row, a_mean_col = a.mean(axis=0), a.mean(axis=1)
    b_mean_row, b_mean_col = b.mean(axis=0), b.mean(axis=1)
    a_mean, b_mean = a.mean(), b.mean()

    A = a - a_mean_row - a_mean_col[:, None] + a_mean
    B = b - b_mean_row - b_mean_col[:, None] + b_mean

    dcov2 = float(np.mean(A * B))
    dvar_x = float(np.mean(A * A))
    dvar_y = float(np.mean(B * B))
    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y)))


def correlation_matrix(
    X: NDArray[np.float64],
    method: CorrelationMethod = "pearson",
) -> NDArray[np.float64]:
    """
    Pairwise correlation matrix for a panel of time series.

    Parameters
    ----------
    X : array (n_series, n_points)
        Panel of univariate series (rows = series).
    method : {"pearson", "spearman", "kendall", "dcor"}
        Correlation measure.

    Returns
    -------
    C : array (n_series, n_series)
        Symmetric correlation matrix with unit diagonal.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n = X.shape[0]
    C = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rho = _pairwise_correlation(X[i], X[j], method)
            C[i, j] = C[j, i] = rho
    return C


def _correlation_to_distance(C: NDArray[np.float64]) -> NDArray[np.float64]:
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    return D


def correlation_network(
    X: NDArray[np.float64],
    method: CorrelationMethod = "pearson",
    rule: NetworkRule = "knn",
    k: int = 5,
    threshold: float = 0.5,
    epsilon: float = 0.3,
    weighted: bool = True,
) -> tuple[nx.Graph, NDArray[np.float64], NDArray[np.float64]]:
    """
    Build a network from pairwise correlations.

    Parameters
    ----------
    X : array (n_series, n_points)
        Panel of time series (nodes = series).
    method : str
        Correlation measure.
    rule : {"knn", "epsilon", "threshold", "complete"}
        How to sparsify the network.
    k : int
        Neighbors for k-NN rule.
    threshold : float
        Minimum |correlation| to keep an edge (``rule="threshold"``).
    epsilon : float
        Maximum correlation distance for epsilon-NN.
    weighted : bool
        Store |correlation| as edge weight.

    Returns
    -------
    G : networkx.Graph
    C : correlation matrix
    D : distance matrix (1 - |C|)
    """
    C = correlation_matrix(X, method=method)
    D = _correlation_to_distance(C)

    if rule == "knn":
        G, _ = net_knn(D, k=validate_positive_int("k", k), weighted=weighted)
    elif rule == "epsilon":
        G, _ = net_enn(D, eps=epsilon, weighted=weighted)
    elif rule == "threshold":
        mask = np.abs(C) >= threshold
        np.fill_diagonal(mask, False)
        D_thr = np.where(mask, D, 0.0)
        G, _ = net_weighted(D_thr, directed=False)
    elif rule == "complete":
        G, _ = net_weighted(D, directed=False)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = float(abs(C[u, v]))

    return G, C, D


def rolling_correlation_matrix(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: CorrelationMethod = "pearson",
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Rolling window correlation between two aligned series.

    Returns
    -------
    values : array (n_windows,)
    centers : array (n_windows,) window center indices
    """
    window = validate_positive_int("window", window)
    step = validate_positive_int("step", step)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    n = len(x)
    if n < window:
        raise ValueError(f"Series length {n} < window {window}")

    values = []
    centers = []
    for start in range(0, n - window + 1, step):
        end = start + window
        values.append(_pairwise_correlation(x[start:end], y[start:end], method))
        centers.append(start + window // 2)

    return np.asarray(values, dtype=np.float64), np.asarray(centers, dtype=np.int64)


def rolling_correlation_network(
    X: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: CorrelationMethod = "pearson",
    rule: NetworkRule = "knn",
    k: int = 3,
    threshold: float = 0.5,
) -> list[tuple[nx.Graph, NDArray[np.float64]]]:
    """
    Sequence of correlation networks over rolling windows.

    Each window uses all series segments ``X[:, t:t+window]``.
    """
    window = validate_positive_int("window", window)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_series, n_points = X.shape
    if n_points < window:
        raise ValueError(f"Series length {n_points} < window {window}")

    results: list[tuple[nx.Graph, NDArray[np.float64]]] = []
    for start in range(0, n_points - window + 1, step):
        segment = X[:, start : start + window]
        G, C, _ = correlation_network(
            segment, method=method, rule=rule, k=k, threshold=threshold
        )
        results.append((G, C))
    return results


def partial_correlation_matrix(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Partial correlation matrix for a multivariate panel.

    Uses the precision matrix formulation (requires ``n_points > n_series``).

    Parameters
    ----------
    X : array (n_series, n_points)
        Panel of time series.

    Returns
    -------
    P : array (n_series, n_series)
        Partial correlation matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n_series, n_points = X.shape
    if n_points <= n_series:
        raise ValueError(
            f"Need n_points > n_series for partial correlation, "
            f"got {n_points} points and {n_series} series"
        )
    return partial_corr(X)


def partial_correlation_network(
    X: NDArray[np.float64],
    rule: NetworkRule = "knn",
    k: int = 5,
    threshold: float = 0.3,
    epsilon: float = 0.5,
    weighted: bool = True,
) -> tuple[nx.Graph, NDArray[np.float64], NDArray[np.float64]]:
    """
    Gaussian graphical-model style network from partial correlations.

    Parameters
    ----------
    X : array (n_series, n_points)
    rule, k, threshold, epsilon, weighted
        Same sparsification options as ``correlation_network``.

    Returns
    -------
    G : networkx.Graph
    P : partial correlation matrix
    D : distance matrix (1 - |P|)
    """
    P = partial_correlation_matrix(X)
    D = _correlation_to_distance(P)

    if rule == "knn":
        G, _ = net_knn(D, k=validate_positive_int("k", k), weighted=weighted)
    elif rule == "epsilon":
        G, _ = net_enn(D, eps=epsilon, weighted=weighted)
    elif rule == "threshold":
        mask = np.abs(P) >= threshold
        np.fill_diagonal(mask, False)
        D_thr = np.where(mask, D, 0.0)
        G, _ = net_weighted(D_thr, directed=False)
    elif rule == "complete":
        G, _ = net_weighted(D, directed=False)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    if weighted:
        for u, v in G.edges():
            G[u][v]["weight"] = float(abs(P[u, v]))

    return G, P, D
