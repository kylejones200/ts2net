"""
Confounder-adjusted causal inference.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .granger import _build_lag_matrix
from ._parallel import pairwise_parallel
from .transfer_entropy import conditional_transfer_entropy


def partial_granger_causality(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    controls: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    max_lag: int = 5,
) -> Dict[str, float]:
    """
    Test whether ``x`` Granger-causes ``y`` after controlling for confounders.

    Uses a nested OLS F-test comparing models with and without ``x`` lags,
    while always including ``y`` lags and control lags.

    Parameters
    ----------
    x, y : array (n,)
        Candidate cause and effect.
    controls : list of arrays or array (n_controls, n)
        Confounding time series.
    max_lag : int, default 5
        Lag order.

    Returns
    -------
    dict
        ``f_stat``, ``p_value``, ``significant``, ``best_lag``, ``n_controls``.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length: {len(x)} != {len(y)}")

    if isinstance(controls, np.ndarray):
        if controls.ndim == 1:
            controls = [controls]
        else:
            controls = [controls[i] for i in range(controls.shape[0])]

    for i, z in enumerate(controls):
        if len(z) != len(y):
            raise ValueError(f"control[{i}] length {len(z)} != y length {len(y)}")

    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    n = len(y)
    if n <= max_lag + 5:
        return {
            "f_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "best_lag": float(max_lag),
            "n_controls": len(controls),
        }

    y_lags = _build_lag_matrix(y, max_lag)
    x_lags = _build_lag_matrix(x, max_lag)
    target = y[max_lag:]
    control_blocks = [_build_lag_matrix(z, max_lag) for z in controls]

    restricted = np.column_stack([y_lags] + control_blocks)
    unrestricted = np.column_stack([y_lags, x_lags] + control_blocks)

    f_stat, p_value = _nested_f_test(restricted, unrestricted, target)

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "best_lag": float(max_lag),
        "n_controls": len(controls),
    }


def conditional_te_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    lag: int = 1,
    bins: int = 10,
    threshold: Optional[float] = None,
    confounder_mode: Literal["aggregate"] = "aggregate",
    series_names: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[nx.DiGraph, NDArray, Dict[str, float]]:
    """
    Build a confounder-adjusted transfer entropy network.

    For each ordered pair ``(i, j)``, conditional TE is computed with a
    confounder series derived from all other variables (aggregated mean).

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Panel of time series.
    lag : int, default 1
        Time lag.
    bins : int, default 10
        Discretization bins.
    threshold : float, optional
        Minimum CTE for retaining an edge.
    confounder_mode : {"aggregate"}, default "aggregate"
        How to form the confounder for each pair (mean of other series).
    series_names : list of str, optional
        Node labels.
    n_jobs : int, default 1
        Parallel workers.

    Returns
    -------
    G : networkx.DiGraph
        Directed network weighted by conditional TE.
    cte_matrix : array (n_series, n_series)
        Conditional TE values.
    stats : dict
        Network summary.
    """
    X, names = _normalize_panel(X, series_names)
    n_series = len(X)

    def _cte_pair(i: int, j: int) -> float:
        others = [X[k] for k in range(n_series) if k not in (i, j)]
        if not others:
            from .transfer_entropy import transfer_entropy

            return transfer_entropy(X[i], X[j], lag=lag, bins=bins)
        z = np.mean(others, axis=0)
        return conditional_transfer_entropy(X[i], X[j], z, lag=lag, bins=bins)

    pair_cte = pairwise_parallel(n_series, _cte_pair, n_jobs=n_jobs)

    cte_matrix = np.zeros((n_series, n_series))
    for (i, j), val in pair_cte.items():
        cte_matrix[i, j] = val

    G = nx.DiGraph()
    G.add_nodes_from(range(n_series))
    for i, name in enumerate(names):
        G.nodes[i]["name"] = name

    for i in range(n_series):
        for j in range(n_series):
            if i == j:
                continue
            val = cte_matrix[i, j]
            if threshold is not None and val < threshold:
                continue
            G.add_edge(i, j, weight=val, adjusted=True)

    stats = {
        "mean_cte": float(np.mean(cte_matrix[cte_matrix > 0]))
        if np.any(cte_matrix > 0)
        else 0.0,
        "max_cte": float(np.max(cte_matrix)),
        "n_edges": G.number_of_edges(),
        "density": G.number_of_edges() / (n_series * (n_series - 1))
        if n_series > 1
        else 0.0,
        "confounder_mode": confounder_mode,
    }
    return G, cte_matrix, stats


def _nested_f_test(
    restricted: NDArray[np.float64],
    unrestricted: NDArray[np.float64],
    target: NDArray[np.float64],
) -> Tuple[float, float]:
    """F-test for nested linear models."""
    n = len(target)

    def rss(design: NDArray[np.float64]) -> Tuple[float, int]:
        x_ = np.column_stack([np.ones(n), design])
        coef, _, _, _ = np.linalg.lstsq(x_, target, rcond=None)
        resid = target - x_ @ coef
        return float(np.sum(resid**2)), x_.shape[1]

    rss_r, k_r = rss(restricted)
    rss_u, k_u = rss(unrestricted)
    df_num = k_u - k_r
    df_den = n - k_u

    if df_num <= 0 or df_den <= 0 or rss_u <= 0:
        return 0.0, 1.0

    f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    f_stat = max(0.0, f_stat)
    p_value = float(stats.f.sf(f_stat, df_num, df_den))
    return f_stat, p_value


def _normalize_panel(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    series_names: Optional[List[str]],
) -> Tuple[List[NDArray[np.float64]], List[str]]:
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = [X]
        elif X.ndim == 2:
            X = [X[i] for i in range(X.shape[0])]
        else:
            raise ValueError(f"X must be 1D or 2D array, got shape {X.shape}")

    n_series = len(X)
    names = series_names or [f"Series_{i}" for i in range(n_series)]
    if len(names) != n_series:
        raise ValueError(
            f"series_names length ({len(names)}) must match "
            f"number of series ({n_series})"
        )
    return X, names
