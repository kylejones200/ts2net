"""
Dynamic Time Warping (DTW) distance.

Preference order for pairwise distance computation:
  1. ts2net_rs.cdist_dtw  — Rust, parallel, Sakoe-Chiba band support
  2. tslearn.metrics.cdist_dtw — C, fast, no band parameter
  3. _dtw_pure_python      — fallback, O(n·m) per pair, slow for large n
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ── backend selection (resolved once at import time) ─────────────────────────

def _select_backend():
    try:
        from ts2net_rs import cdist_dtw as _rs
        return "rust", _rs
    except ImportError:
        pass
    try:
        from tslearn.metrics import cdist_dtw as _ts
        return "tslearn", _ts
    except ImportError:
        pass
    return "python", None


_BACKEND, _BACKEND_FN = _select_backend()


# ── public API ────────────────────────────────────────────────────────────────

def cdist_dtw(
    X: NDArray[np.float64],
    band: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Pairwise DTW distance matrix for a collection of time series.

    Parameters
    ----------
    X : ndarray of shape (n_series, n_timesteps)
        Each row is one time series.
    band : int, optional
        Sakoe-Chiba bandwidth (maximum allowed warping). ``None`` means
        unconstrained warping. Only honoured by the Rust backend; tslearn
        and the pure-Python fallback ignore it silently.

    Returns
    -------
    D : ndarray of shape (n_series, n_series)
        Symmetric distance matrix; diagonal is zero.

    Notes
    -----
    Backend used: ``ts2net.distances.dtw._BACKEND`` is one of
    ``"rust"``, ``"tslearn"``, or ``"python"``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_series × n_timesteps), got shape {X.shape}")

    if _BACKEND == "rust":
        return np.asarray(_BACKEND_FN(X, band=band), dtype=np.float64)

    if _BACKEND == "tslearn":
        return np.asarray(_BACKEND_FN(X), dtype=np.float64)

    # pure-Python fallback
    return _cdist_python(X)


def dtw_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """
    DTW distance between two time series (pairwise scalar).

    For batch computation prefer ``cdist_dtw`` which uses the Rust backend.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


# ── internal ──────────────────────────────────────────────────────────────────

def _cdist_python(X: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(X[i], X[j])
            D[i, j] = D[j, i] = d
    return D
