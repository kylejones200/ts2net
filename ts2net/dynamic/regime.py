"""
Structural break detection on graph metric time series.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def detect_regime_changes(
    values: NDArray[np.float64],
    method: Literal["zscore", "cusum"] = "zscore",
    threshold: float = 2.5,
    min_segment: int = 3,
) -> dict[str, NDArray[np.int64] | NDArray[np.float64] | list[int]]:
    """
    Detect regime changes in a sequence of graph metrics.

    Parameters
    ----------
    values : array (n_windows,)
        Metric sampled at each graph window (e.g. average degree).
    method : {"zscore", "cusum"}, default "zscore"
        ``zscore`` flags large jumps in first differences.
        ``cusum`` uses a cumulative-sum deviation test.
    threshold : float, default 2.5
        Detection threshold (z-score or normalized CUSUM scale).
    min_segment : int, default 3
        Minimum windows between consecutive breaks.

    Returns
    -------
    dict
        ``break_indices``, ``scores``, ``segments`` (start indices per regime).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 3:
        return {
            "break_indices": np.array([], dtype=np.int64),
            "scores": np.zeros(n, dtype=np.float64),
            "segments": [0] if n else [],
        }

    if method == "zscore":
        diff = np.diff(values)
        mu = np.mean(diff)
        sigma = np.std(diff)
        if sigma < 1e-12:
            scores = np.zeros(n, dtype=np.float64)
        else:
            z = np.abs((diff - mu) / sigma)
            scores = np.concatenate([[0.0], z])
    else:
        centered = values - np.mean(values)
        cusum = np.cumsum(centered)
        scale = np.std(cusum) if np.std(cusum) > 1e-12 else 1.0
        scores = np.abs(cusum) / scale

    candidates = np.where(scores >= threshold)[0].tolist()
    breaks: list[int] = []
    last = -min_segment
    for idx in candidates:
        if idx - last >= min_segment:
            breaks.append(int(idx))
            last = idx

    segments = [0] + breaks
    if segments[-1] != n:
        segments.append(n)

    return {
        "break_indices": np.asarray(breaks, dtype=np.int64),
        "scores": scores.astype(np.float64),
        "segments": segments,
    }
