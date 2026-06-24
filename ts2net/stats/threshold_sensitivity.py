"""
Threshold sensitivity sweeps for network builders (Horizon 9 / v0.9).
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..api import NVG
from ..core.recurrence import RecurrenceNetwork


def _delay_embed(x: NDArray[np.float64], m: int = 5, tau: int = 1) -> NDArray[np.float64]:
    n = len(x) - (m - 1) * tau
    if n < 2:
        raise ValueError("series too short for delay embedding")
    return np.column_stack([x[i * tau : i * tau + n] for i in range(m)])


def _graph_stats(G: Any) -> dict[str, float]:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = (2.0 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
    degrees = np.array([d for _, d in G.degree()], dtype=np.float64)
    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": float(density),
        "avg_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
    }


def threshold_sensitivity_sweep(
    x: NDArray[np.float64],
    method: Literal["recurrence", "nvg", "correlation"] = "recurrence",
    thresholds: Sequence[float] | None = None,
    *,
    rule: str = "epsilon",
    embed_dim: int = 5,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Sweep a threshold parameter and record network summary statistics.

    Parameters
    ----------
    x : array
        Univariate time series.
    method : {"recurrence", "nvg", "correlation"}
        Builder family to evaluate.
    thresholds : sequence of float, optional
        Threshold grid. Defaults depend on ``method``.
    rule : str, default "epsilon"
        Recurrence rule (``epsilon`` or ``knn``). Ignored for other methods.
    limit : int, optional
        NVG visibility horizon limit.
    **kwargs
        Extra builder arguments.

    Returns
    -------
    pandas.DataFrame
        One row per threshold with ``threshold`` and graph statistics.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if thresholds is None:
        if method == "recurrence":
            thresholds = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        elif method == "correlation":
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            thresholds = [10, 25, 50, 100, 200]

    rows: list[dict[str, float]] = []
    for thr in thresholds:
        if method == "recurrence":
            if rule == "knn":
                rn = RecurrenceNetwork(rule="knn", k=int(thr), **kwargs)
            else:
                rn = RecurrenceNetwork(rule="epsilon", threshold=float(thr), **kwargs)
            G, _ = rn.fit_transform(x)
            stats = _graph_stats(G)
            stats["threshold"] = float(thr)
            rows.append(stats)
        elif method == "nvg":
            builder = NVG(limit=int(thr), output="stats", **kwargs).build(x)
            stats = builder.stats()
            stats["threshold"] = float(thr)
            rows.append(stats)
        elif method == "correlation":
            from ..graphs.correlation import correlation_network

            emb = _delay_embed(x, m=embed_dim)
            G, _, _ = correlation_network(
                emb, rule="threshold", threshold=float(thr), **kwargs
            )
            stats = _graph_stats(G)
            stats["threshold"] = float(thr)
            rows.append(stats)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
