"""
Windowed graphs API for meter data and large time series.

Provides high-level API for building graph statistics per window,
storing only time series of stats (not full graphs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .factory import aggregate_stats
from .multivariate.windows import ts_to_windows
from .scale.streaming import (
    _make_window_config,
    _stats_for_window,
    iter_windows,
)


def _empty_stats() -> dict[str, float]:
    return {
        "n_nodes": 0,
        "n_edges": 0,
        "avg_degree": float("nan"),
        "std_degree": float("nan"),
    }


def _compute_window_stats(
    window_data: NDArray[np.float64],
    method: str,
    config: object,
    aggregate: str | None,
) -> dict[str, float] | float:
    try:
        stats = _stats_for_window(window_data, method, config)
        if aggregate:
            return aggregate_stats(stats, aggregate)
        return stats
    except Exception:
        if aggregate:
            return float("nan")
        return _empty_stats()


def build_windows(
    x: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: str = "hvg",
    output: str = "stats",
    aggregate: str | None = None,
    n_jobs: int = 1,
    streaming: bool = False,
    **method_kwargs,
) -> dict[str, np.ndarray] | np.ndarray:
    """
    Build graph statistics per window (memory efficient for large series).

    Parameters
    ----------
    x : array (n_points,)
        Input time series
    window : int
        Window width (number of time points per window)
    step : int, default 1
        Step size between consecutive windows
    method : str, default "hvg"
        Network method: 'hvg', 'nvg', 'recurrence', 'transition'
    output : str, default "stats"
        Output mode: 'stats' (recommended), 'degrees', or 'edges'
    aggregate : str, optional
        Aggregation function for stats: 'mean', 'std', 'min', 'max'
    n_jobs : int, default 1
        Parallel workers for independent windows. Use -1 for all CPUs.
    streaming : bool, default False
        If True, avoid materializing the full ``(n_windows, window)`` matrix.
    **method_kwargs
        Additional parameters for the network builder

    Returns
    -------
    dict[str, np.ndarray] or np.ndarray
        Per-window stats arrays, or a single array when ``aggregate`` is set.
    """
    method_key = method.lower()
    config = _make_window_config(method_key, window, output, method_kwargs)

    if streaming:
        window_iter = (
            (i, w) for i, _, w in iter_windows(x, window, step)
        )
    else:
        windows = ts_to_windows(x, width=window, by=step)
        window_iter = ((i, windows[i]) for i in range(windows.shape[0]))

    window_list = list(window_iter)
    n_windows = len(window_list)

    if n_jobs != 1 and n_windows > 1:
        from joblib import Parallel, delayed

        computed = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_window_stats)(w, method_key, config, aggregate)
            for _, w in window_list
        )
    else:
        computed = [
            _compute_window_stats(w, method_key, config, aggregate)
            for _, w in window_list
        ]

    if aggregate:
        out = np.zeros(n_windows, dtype=np.float64)
        for i, val in enumerate(computed):
            out[i] = float(val)  # type: ignore[arg-type]
        return out

    result = {
        "n_nodes": np.zeros(n_windows, dtype=np.int64),
        "n_edges": np.zeros(n_windows, dtype=np.int64),
        "avg_degree": np.zeros(n_windows, dtype=np.float64),
        "std_degree": np.zeros(n_windows, dtype=np.float64),
    }
    for i, stats in enumerate(computed):
        s = stats  # type: ignore[assignment]
        result["n_nodes"][i] = int(s["n_nodes"])
        result["n_edges"][i] = int(s["n_edges"])
        result["avg_degree"][i] = float(s["avg_degree"])
        result["std_degree"][i] = float(s.get("std_degree", 0.0))
    return result
