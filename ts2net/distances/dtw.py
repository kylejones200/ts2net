"""
Dynamic Time Warping (DTW) distance.

Preference order for pairwise distance computation (``auto`` / default):
  1. ts2net_rs.cdist_dtw  — Rust, parallel, Sakoe-Chiba band support
  2. tslearn.metrics.cdist_dtw — C, fast, no band parameter
  3. _dtw_pure_python      — fallback, O(n·m) per pair, slow for large n

Set ``TS2NET_BACKEND`` or pass ``backend=`` to override.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..core.backend import resolve_compute_backend

# Panels larger than this use block-wise Rust DTW to cap peak memory.
_DEFAULT_CHUNK_SIZE = 256
_DEFAULT_PANEL_CHUNK_THRESHOLD = 64


def _select_backend(requested: str = "auto") -> tuple[str, object | None]:
    """Map unified backend names to a DTW implementation."""
    resolved = resolve_compute_backend(requested)
    if resolved == "rust":
        try:
            from ts2net_rs import cdist_dtw as _rs

            return "rust", _rs
        except ImportError:
            pass
    if resolved in ("rust", "numba"):
        try:
            from tslearn.metrics import cdist_dtw as _ts

            return "tslearn", _ts
        except ImportError:
            pass
    return "python", None


_BACKEND, _BACKEND_FN = _select_backend("auto")


def get_dtw_backend(requested: str = "auto") -> str:
    """Return active DTW backend label: ``rust``, ``tslearn``, or ``python``."""
    name, _ = _select_backend(requested)
    return name


def cdist_dtw(
    X: NDArray[np.float64],
    band: Optional[int] = None,
    *,
    backend: str = "auto",
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
    backend : str, default ``auto``
        ``rust`` | ``numba`` (tslearn) | ``python``.

    Returns
    -------
    D : ndarray of shape (n_series, n_series)
        Symmetric distance matrix; diagonal is zero.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_series × n_timesteps), got shape {X.shape}")

    name, fn = _select_backend(backend)

    if name == "rust":
        return np.asarray(fn(X, band=band), dtype=np.float64)

    if name == "tslearn":
        return np.asarray(fn(X), dtype=np.float64)

    return _cdist_python(X)


def cdist_dtw_chunked(
    X: NDArray[np.float64],
    band: Optional[int] = None,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    backend: str = "auto",
) -> NDArray[np.float64]:
    """
    Block-wise DTW distance matrix for large panels (Rust rectangular kernels).

    Falls back to :func:`cdist_dtw` when the panel fits in one chunk or Rust
    is unavailable.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n <= chunk_size:
        return cdist_dtw(X, band=band, backend=backend)

    name, _ = _select_backend(backend)
    if name != "rust":
        return cdist_dtw(X, band=band, backend=backend)

    try:
        from ts2net_rs import cdist_dtw as _cdist_block
        from ts2net_rs import cdist_dtw_rectangular as _cdist_rect
    except ImportError:
        return cdist_dtw(X, band=band, backend=backend)

    D = np.zeros((n, n), dtype=np.float64)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        block = _cdist_block(X[i0:i1], band=band)
        D[i0:i1, i0:i1] = np.asarray(block, dtype=np.float64)
        for j0 in range(i1, n, chunk_size):
            j1 = min(j0 + chunk_size, n)
            rect = np.asarray(
                _cdist_rect(X[i0:i1], X[j0:j1], band=band), dtype=np.float64
            )
            D[i0:i1, j0:j1] = rect
            D[j0:j1, i0:i1] = rect.T
    return D


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


def _cdist_python(X: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(X[i], X[j])
            D[i, j] = D[j, i] = d
    return D
