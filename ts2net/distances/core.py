"""
Core distance metrics for time series analysis.

This module implements various distance metrics for comparing time series,
including correlation-based, dynamic time warping, and information-theoretic measures.
"""

import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from scipy import stats
from scipy.spatial.distance import squareform, pdist
from scipy.signal import correlate

# Optional deps
try:
    from tslearn.metrics import cdist_dtw as _cdist_dtw
except ImportError:
    _cdist_dtw = None

try:
    import minepy
except ImportError:
    minepy = None


def tsdist_cor(
    X: np.ndarray, method: str = "pearson", absolute: bool = False
) -> np.ndarray:
    """
    Compute distance matrix using correlation as a distance measure.

    Args:
        X: Input time series array of shape (n_series, n_timesteps)
        method: Correlation method ('pearson' or 'spearman')
        absolute: If True, use absolute value of correlation

    Returns:
        Distance matrix of shape (n_series, n_series)
    """
    if method == "pearson":
        corr = np.corrcoef(X)
    elif method == "spearman":
        corr, _ = stats.spearmanr(X, axis=1)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    if absolute:
        corr = np.abs(corr)

    # Convert correlation to distance (0-2 range)
    return np.sqrt(2 * (1 - corr))


def tsdist_ccf(X: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute distance matrix using maximum cross-correlation.

    Args:
        X: Input time series array of shape (n_series, n_timesteps)
        max_lag: Maximum lag to consider for cross-correlation

    Returns:
        Distance matrix of shape (n_series, n_series)
    """
    n = len(X)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            x, y = X[i], X[j]
            ccf = correlate(x - x.mean(), y - y.mean(), mode="full")
            lags = np.arange(-(len(x) - 1), len(x))
            mask = (lags >= -max_lag) & (lags <= max_lag)
            max_r = np.max(np.abs(ccf[mask])) / (np.std(x) * np.std(y) * len(x))
            D[i, j] = D[j, i] = np.sqrt(2 * (1 - max_r))

    return D


def tsdist_dtw(X: np.ndarray) -> np.ndarray:
    """
    Compute Dynamic Time Warping (DTW) distance matrix.

    Args:
        X: Input time series array of shape (n_series, n_timesteps)

    Returns:
        Distance matrix of shape (n_series, n_series)
    """
    if _cdist_dtw is not None:
        # Use optimized tslearn implementation if available
        return _cdist_dtw(X)

    # Fallback implementation
    from .dtw import dtw_distance

    n = len(X)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(X[i], X[j])
            D[i, j] = D[j, i] = d

    return D


def _entropy_from_counts(c: np.ndarray) -> float:
    """Calculate entropy from count array."""
    p = c / np.sum(c)
    return -np.sum(p * np.log2(p + 1e-10))


def tsdist_nmi(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Compute normalized mutual information between two time series.

    Args:
        x, y: Input time series
        bins: Number of bins for discretization

    Returns:
        Normalized mutual information distance (0-1 range)
    """
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    c_x = np.sum(c_xy, axis=1)
    c_y = np.sum(c_xy, axis=0)

    hx = _entropy_from_counts(c_x)
    hy = _entropy_from_counts(c_y)
    hxy = _entropy_from_counts(c_xy)

    # Normalized mutual information
    nmi = 2 * (hx + hy - hxy) / (hx + hy)
    return 1 - nmi  # Convert to distance


def tsdist_voi(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Compute variation of information between two time series.

    Args:
        x, y: Input time series
        bins: Number of bins for discretization

    Returns:
        Variation of information distance
    """
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    c_x = np.sum(c_xy, axis=1)
    c_y = np.sum(c_xy, axis=0)

    hx = _entropy_from_counts(c_x)
    hy = _entropy_from_counts(c_y)
    hxy = _entropy_from_counts(c_xy)

    # Variation of information
    return hx + hy - 2 * hxy


def tsdist_mic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Maximal Information Coefficient (MIC) between two time series.

    Args:
        x, y: Input time series

    Returns:
        MIC distance (0-1 range)
    """
    if minepy is not None:
        # Use minepy if available
        mic = minepy.MINE()
        mic.compute_score(x, y)
        return 1 - mic.mic()

    # Fallback implementation (simplified)
    return tsdist_nmi(x, y, bins=min(32, int(np.sqrt(len(x)))))


def tsdist_vr(
    t1: np.ndarray, t2: np.ndarray, tau: float = 1.0, T: Optional[float] = None
) -> float:
    """
    Compute Van Rossum distance between two spike trains.

    Args:
        t1, t2: Arrays of spike times
        tau: Time constant for exponential kernel
        T: Total time interval (if None, use max spike time)

    Returns:
        Van Rossum distance
    """
    if T is None:
        T = (
            max(np.max(t1) if len(t1) > 0 else 0, np.max(t2) if len(t2) > 0 else 0)
            + 5 * tau
        )

    def spike_kernel(t, spikes):
        result = np.zeros_like(t)
        for s in spikes:
            mask = t >= s
            result[mask] += np.exp(-(t[mask] - s) / tau)
        return result

    t = np.linspace(0, T, 1000)
    s1 = spike_kernel(t, t1)
    s2 = spike_kernel(t, t2)

    return np.sqrt(np.trapz((s1 - s2) ** 2, t)) / np.sqrt(tau / 2)


def dist_percentile(D: np.ndarray, q: float) -> float:
    """
    Compute a percentile of the upper triangle of a distance matrix.

    Args:
        D: Distance matrix
        q: Percentile (0-100)

    Returns:
        The q-th percentile of the upper triangle values
    """
    return np.percentile(D[np.triu_indices_from(D, k=1)], q)


def dist_matrix_normalize(D: np.ndarray, kind: str = "minmax") -> np.ndarray:
    """
    Normalize a distance matrix.

    Args:
        D: Input distance matrix
        kind: Normalization method ('minmax' or 'zscore')

    Returns:
        Normalized distance matrix
    """
    if kind == "minmax":
        D_min, D_max = D.min(), D.max()
        if D_max > D_min:
            return (D - D_min) / (D_max - D_min)
        else:
            return np.zeros_like(D)
    elif kind == "zscore":
        D_mean, D_std = D.mean(), D.std()
        if D_std > 0:
            return (D - D_mean) / D_std
        else:
            return np.zeros_like(D)
    else:
        raise ValueError(f"Unknown normalization kind: {kind}")


# Wrapper functions for compatibility with main __init__.py imports
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute DTW distance between two time series.
    
    Args:
        x, y: Input time series
        
    Returns:
        DTW distance
    """
    X = np.array([x, y])
    D = tsdist_dtw(X)
    return D[0, 1]


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Euclidean distance between two time series.
    
    Args:
        x, y: Input time series
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(x - y)


def correlation_distance(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    """
    Compute correlation-based distance between two time series.
    
    Args:
        x, y: Input time series
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation distance (1 - |correlation|)
    """
    X = np.array([x, y])
    D = tsdist_cor(X, method=method, absolute=True)
    return D[0, 1]
