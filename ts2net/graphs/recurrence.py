"""
Extended recurrence network builders.

Adaptive threshold selection and cross-recurrence between two series.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from .._validation import validate_positive_int, validate_series
from ..api import RecurrenceNetwork


def _pairwise_distance_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return squareform(pdist(x.reshape(-1, 1), metric="euclidean"))


def _epsilon_for_density(D: NDArray[np.float64], target_density: float) -> float:
    """Choose epsilon so recurrence density ≈ target_density."""
    target_density = float(np.clip(target_density, 1e-6, 1.0))
    n = D.shape[0]
    upper = D[np.triu_indices(n, k=1)]
    if len(upper) == 0:
        return 0.1
    # density ≈ 2m / (n(n-1))  =>  m ≈ density * n(n-1)/2
    target_edges = target_density * n * (n - 1) / 2
    k = int(np.clip(target_edges, 1, len(upper)))
    return float(np.partition(upper, k - 1)[k - 1])


def adaptive_recurrence_network(
    x: NDArray[np.float64],
    target_density: float = 0.05,
    m: int | None = None,
    tau: int = 1,
    metric: str = "euclidean",
    output: str = "edges",
) -> tuple[RecurrenceNetwork, float]:
    """
    Recurrence network with epsilon chosen for a target edge density.

    Parameters
    ----------
    x : array (n,)
        Input series.
    target_density : float, default 0.05
        Desired edge density (fraction of possible pairs).
    m, tau : int
        Embedding dimension and delay (``m=None`` uses raw series).
    metric : str
        Distance metric for recurrence.

    Returns
    -------
    builder : RecurrenceNetwork
        Fitted builder.
    epsilon : float
        Selected threshold.
    """
    x = validate_series(x, "adaptive_recurrence_network")
    tau = validate_positive_int("tau", tau)

    D = _pairwise_distance_matrix(x)
    epsilon = _epsilon_for_density(D, target_density)

    builder = RecurrenceNetwork(
        m=m,
        tau=tau,
        rule="epsilon",
        epsilon=epsilon,
        metric=metric,
        output=output,
    ).build(x)
    return builder, epsilon


def cross_recurrence_network(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    epsilon: float | None = None,
    target_density: float = 0.05,
) -> tuple[nx.Graph, NDArray[np.bool_]]:
    """
    Cross-recurrence network between two time series.

    Node ``i`` connects to node ``j`` when ``|x[i]-y[j]| < epsilon`` (bivariate
    recurrence in index-aligned embedding).

    Parameters
    ----------
    x, y : array (n,)
        Aligned time series.
    epsilon : float, optional
        Threshold; if None, chosen from ``target_density``.
    target_density : float
        Used when ``epsilon`` is None.

    Returns
    -------
    G : networkx.Graph
    R : array (n, n) bool cross-recurrence matrix
    """
    x = validate_series(x, "cross_recurrence_network")
    y = validate_series(y, "cross_recurrence_network")
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")

    n = len(x)
    diff = np.abs(x[:, None] - y[None, :])
    if epsilon is None:
        off_diag = diff[~np.eye(n, dtype=bool)]
        epsilon = float(np.percentile(off_diag, 100 * (1 - target_density)))

    R = diff <= epsilon
    np.fill_diagonal(R, False)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(np.triu(R, k=1))
    G.add_edges_from(zip(rows.tolist(), cols.tolist(), strict=True))
    return G, R


def recurrence_matrix(
    x: NDArray[np.float64],
    epsilon: float | None = None,
    target_density: float = 0.05,
) -> NDArray[np.bool_]:
    """
    Boolean recurrence matrix for a univariate series.

    Parameters
    ----------
    x : array (n,)
    epsilon : float, optional
        Distance threshold; chosen from ``target_density`` if None.
    """
    x = validate_series(x, "recurrence_matrix")
    D = _pairwise_distance_matrix(x)
    n = len(x)

    if epsilon is None:
        epsilon = _epsilon_for_density(D, target_density)

    R = D <= epsilon
    np.fill_diagonal(R, False)
    return R


def recurrence_quantification(
    x: NDArray[np.float64],
    epsilon: float | None = None,
    target_density: float = 0.05,
    m: int | None = None,
    tau: int = 1,
    lmin: int = 2,
    vmin: int = 2,
    output: str = "edges",
) -> dict:
    """
    Recurrence network plus RQA (recurrence quantification analysis) metrics.

    Parameters
    ----------
    x : array (n,)
        Input series.
    epsilon : float, optional
        Recurrence threshold.
    target_density : float
        Used to pick ``epsilon`` when not provided.
    m, tau : int
        Embedding parameters passed to the recurrence builder.
    lmin, vmin : int
        Minimum diagonal / vertical line lengths for RQA.
    output : str
        Builder output mode.

    Returns
    -------
    dict
        Keys: ``epsilon``, ``rqa``, ``recurrence_rate``, ``builder``, ``matrix``.
        ``rqa`` contains RR, DET, L, Lmax, ENTR, LAM, TT.
    """
    from ..stats.stats import rqa_full

    x = validate_series(x, "recurrence_quantification")
    tau = validate_positive_int("tau", tau)

    if epsilon is None:
        builder, epsilon = adaptive_recurrence_network(
            x,
            target_density=target_density,
            m=m,
            tau=tau,
            output=output,
        )
    else:
        builder = RecurrenceNetwork(
            m=m,
            tau=tau,
            rule="epsilon",
            epsilon=epsilon,
            output=output,
        ).build(x)

    R = recurrence_matrix(x, epsilon=epsilon)
    A = R.astype(np.float64)
    rqa = rqa_full(A, lmin=lmin, vmin=vmin)

    return {
        "epsilon": float(epsilon),
        "rqa": rqa,
        "recurrence_rate": rqa["RR"],
        "builder": builder,
        "matrix": R,
    }
