"""
Network anomaly detection on rolling graph sequences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _zscore(x: NDArray[np.float64]) -> NDArray[np.float64]:
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma


def window_anomaly_scores(
    stats: dict[str, NDArray[np.float64]],
    metrics: list[str] | None = None,
) -> NDArray[np.float64]:
    """
    Per-window anomaly score from graph summary statistics.

    Uses the maximum absolute z-score across selected metrics at each window.

    Parameters
    ----------
    stats : dict[str, array]
        Window-level stats (e.g. from ``RollingGraphSequence.stats`` or
        ``build_windows`` output).
    metrics : list of str, optional
        Metrics to include. Defaults to numeric keys with length > 2.

    Returns
    -------
    array (n_windows,)
        Anomaly score per window (higher = more unusual).
    """
    if metrics is None:
        metrics = [
            k
            for k, v in stats.items()
            if isinstance(v, np.ndarray) and v.dtype.kind in "fi" and len(v) > 2
        ]

    if not metrics:
        n = len(next(iter(stats.values()))) if stats else 0
        return np.zeros(n, dtype=np.float64)

    z_cols = []
    for key in metrics:
        arr = np.asarray(stats[key], dtype=np.float64)
        z_cols.append(np.abs(_zscore(arr)))
    return np.max(np.vstack(z_cols), axis=0)


def edge_transition_anomalies(
    births: NDArray[np.float64],
    deaths: NDArray[np.float64],
    jaccard: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Anomaly scores for graph transitions (length ``n_windows - 1``).

    Combines z-scored edge births, deaths, and optional Jaccard drop.
    """
    births = np.asarray(births, dtype=np.float64)
    deaths = np.asarray(deaths, dtype=np.float64)
    if len(births) == 0:
        return np.array([], dtype=np.float64)

    parts = [np.abs(_zscore(births)), np.abs(_zscore(deaths))]
    if jaccard is not None:
        jaccard = np.asarray(jaccard, dtype=np.float64)
        parts.append(np.abs(_zscore(1.0 - jaccard)))
    return np.max(np.vstack(parts), axis=0)
