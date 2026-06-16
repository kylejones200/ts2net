"""
Soft-DTW and matrix-profile similarity distances.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int


def _znorm_subsequences(x: NDArray[np.float64], subseq_len: int) -> NDArray[np.float64]:
    n_windows = len(x) - subseq_len + 1
    out = np.empty((n_windows, subseq_len), dtype=np.float64)
    for i in range(n_windows):
        seg = x[i : i + subseq_len]
        std = seg.std()
        out[i] = (seg - seg.mean()) / std if std > 1e-12 else seg - seg.mean()
    return out


def matrix_profile_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    subseq_len: int = 10,
) -> float:
    """
    Mean minimum z-normalized subsequence distance (matrix-profile style).

    For each subsequence in ``x``, finds the closest subsequence in ``y`` and
    returns the average min distance. Lower values indicate stronger shape
    similarity.

    Parameters
    ----------
    x, y : array (n,)
        Input series.
    subseq_len : int
        Subsequence / motif length.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    subseq_len = validate_positive_int("subseq_len", subseq_len)

    if len(x) < subseq_len or len(y) < subseq_len:
        return 1.0

    Sx = _znorm_subsequences(x, subseq_len)
    Sy = _znorm_subsequences(y, subseq_len)
    mins = [float(np.linalg.norm(sx - Sy, axis=1).min()) for sx in Sx]
    return float(np.mean(mins))


def matrix_profile_distance_matrix(
    X: NDArray[np.float64],
    subseq_len: int = 10,
) -> NDArray[np.float64]:
    """Pairwise matrix-profile distances for a panel."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = matrix_profile_distance(X[i], X[j], subseq_len=subseq_len)
            D[i, j] = D[j, i] = d
    return D


def soft_dtw_distance(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    gamma: float = 1.0,
) -> float:
    """
    Soft-DTW dissimilarity between two series.

    Uses ``tslearn`` when installed; falls back to standard DTW otherwise.

    Parameters
    ----------
    gamma : float
        Smoothing parameter (smaller = closer to hard DTW).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    try:
        from tslearn.metrics import soft_dtw

        return float(
            soft_dtw(
                x.reshape(-1, 1),
                y.reshape(-1, 1),
                gamma=gamma,
            )
        )
    except ImportError:
        from ..multivariate.distances import tsdist_dtw

        return float(tsdist_dtw(x, y))


def soft_dtw_distance_matrix(
    X: NDArray[np.float64],
    gamma: float = 1.0,
) -> NDArray[np.float64]:
    """Pairwise soft-DTW distance matrix."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = soft_dtw_distance(X[i], X[j], gamma=gamma)
            D[i, j] = D[j, i] = d
    return D
