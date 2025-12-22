"""
Distance functions for comparing multiple time series.

Implements R ts2net API for distance matrix calculation.

API Design Credit
-----------------
The distance functions and API design are based on the R ts2net package by
Leonardo N. Ferreira:

    Ferreira, L.N. (2024). From time series to networks in R with the ts2net 
    package. Applied Network Science, 9(1), 32.
    https://doi.org/10.1007/s41109-024-00642-2
    
Original R package: https://github.com/lnferreira/ts2net

This Python implementation extends the R API with:
- Numba acceleration for performance
- Parallel processing with joblib
- Additional distance functions
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable, Dict
import logging
from scipy.stats import pearsonr
from scipy.signal import correlate
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


# ============================================================================
# Distance Functions
# ============================================================================

def tsdist_cor(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation distance: d = 1 - |ρ|"""
    if len(x) != len(y):
        raise ValueError("Time series must have same length")
    rho = np.corrcoef(x, y)[0, 1]
    return 1.0 - abs(rho)


def tsdist_ccf(x: np.ndarray, y: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Cross-correlation distance: d = 1 - max(|CCF|)"""
    if len(x) != len(y):
        raise ValueError("Time series must have same length")
    
    # Normalize
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)
    
    # Calculate cross-correlation
    ccf = correlate(x_norm, y_norm, mode='full')
    ccf /= len(x)
    
    # Restrict to max_lag
    if max_lag is not None:
        center = len(ccf) // 2
        ccf = ccf[center - max_lag:center + max_lag + 1]
    
    return 1.0 - np.max(np.abs(ccf))


def tsdist_dtw(x: np.ndarray, y: np.ndarray, window: Optional[int] = None, 
               normalize: bool = True) -> float:
    """Dynamic Time Warping distance"""
    if HAS_NUMBA:
        return _dtw_numba(x, y, window, normalize)
    return _dtw_python(x, y, window, normalize)


@njit(cache=True)
def _dtw_numba(x: np.ndarray, y: np.ndarray, window: Optional[int], 
               normalize: bool) -> float:
    """Numba-accelerated DTW"""
    n, m = len(x), len(y)
    
    # Initialize cost matrix with infinity
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    
    # Fill cost matrix
    for i in range(1, n + 1):
        # Apply Sakoe-Chiba band if window specified
        j_start = max(1, i - window) if window is not None else 1
        j_end = min(m + 1, i + window + 1) if window is not None else m + 1
        
        for j in range(j_start, j_end):
            dist = abs(x[i - 1] - y[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    
    distance = cost[n, m]
    
    if normalize:
        distance /= (n + m)
    
    return distance


def _dtw_python(x: np.ndarray, y: np.ndarray, window: Optional[int], 
                normalize: bool) -> float:
    """Python fallback for DTW"""
    n, m = len(x), len(y)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    
    for i in range(1, n + 1):
        j_start = max(1, i - window) if window is not None else 1
        j_end = min(m + 1, i + window + 1) if window is not None else m + 1
        
        for j in range(j_start, j_end):
            dist = abs(x[i - 1] - y[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    
    distance = cost[n, m]
    
    if normalize:
        distance /= (n + m)
    
    return distance


def tsdist_nmi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Normalized Mutual Information distance: d = 1 - NMI"""
    # Discretize
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins + 1)[1:-1])
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), bins + 1)[1:-1])
    
    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(x_binned, y_binned, bins=bins)
    
    # Probabilities
    pxy = hist_2d / len(x)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropies
    px_py = px[:, None] * py[None, :]
    
    # Avoid log(0)
    mask = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / px_py[mask]))
    
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    
    nmi = 2 * mi / (hx + hy) if (hx + hy) > 0 else 0.0
    
    return 1.0 - nmi


def tsdist_es(x: np.ndarray, y: np.ndarray, threshold: Optional[float] = None, 
              tau: int = 1) -> float:
    """Event Synchronization distance"""
    # Detect events (peaks above threshold)
    if threshold is None:
        threshold = np.median(x)
    
    events_x = np.where(x > threshold)[0]
    events_y = np.where(y > threshold)[0]
    
    if len(events_x) == 0 or len(events_y) == 0:
        return 1.0
    
    # Count synchronized events
    sync_count = 0
    for ex in events_x:
        for ey in events_y:
            if abs(ex - ey) <= tau:
                sync_count += 1
                break
    
    # Normalize
    sync = 2 * sync_count / (len(events_x) + len(events_y))
    
    return 1.0 - sync


def tsdist_voi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """
    Variation of Information distance.
    
    VOI = H(X) + H(Y) - 2*MI(X,Y) where H is entropy and MI is mutual information.
    
    Parameters
    ----------
    x, y : array
        Time series to compare
    bins : int
        Number of bins for discretization
    
    Returns
    -------
    float
        VOI distance (0 = identical, higher = more different)
    """
    # Discretize
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), bins + 1)[1:-1])
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), bins + 1)[1:-1])
    
    # Calculate joint histogram
    hist_2d, _, _ = np.histogram2d(x_binned, y_binned, bins=bins)
    
    # Probabilities
    pxy = hist_2d / len(x)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropies
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    
    # Mutual information
    px_py = px[:, None] * py[None, :]
    mask = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[mask] * np.log(pxy[mask] / px_py[mask]))
    
    # VOI = H(X) + H(Y) - 2*MI(X,Y)
    voi = hx + hy - 2 * mi
    
    return voi


def tsdist_mic(x: np.ndarray, y: np.ndarray, alpha: float = 0.6, c: int = 15) -> float:
    """
    Maximal Information Coefficient distance: d = 1 - MIC.
    
    MIC measures general association strength between variables.
    Requires the `minepy` package.
    
    Parameters
    ----------
    x, y : array
        Time series to compare
    alpha : float
        MIC parameter controlling search grid resolution
    c : int
        MIC parameter controlling grid complexity
    
    Returns
    -------
    float
        MIC-based distance (0 = perfect association, 1 = no association)
    
    Notes
    -----
    Install minepy: pip install minepy
    """
    try:
        from minepy import MINE
    except ImportError:
        raise ImportError(
            "tsdist_mic requires the 'minepy' package. "
            "Install it with: pip install minepy"
        )
    
    mine = MINE(alpha=alpha, c=c)
    mine.compute_score(x, y)
    mic = mine.mic()
    
    return 1.0 - mic


def tsdist_vr(x: np.ndarray, y: np.ndarray, tau: float = 1.0, 
              threshold: Optional[float] = None) -> float:
    """
    van Rossum spike train distance.
    
    Treats time series as spike trains and compares them using
    exponentially decaying kernels.
    
    Parameters
    ----------
    x, y : array
        Time series to compare
    tau : float
        Exponential kernel time constant
    threshold : float, optional
        Spike detection threshold (None = median)
    
    Returns
    -------
    float
        van Rossum distance (0 = identical spike trains)
    """
    # Detect spikes (events above threshold)
    if threshold is None:
        threshold_x = np.median(x)
        threshold_y = np.median(y)
    else:
        threshold_x = threshold_y = threshold
    
    spikes_x = np.where(x > threshold_x)[0]
    spikes_y = np.where(y > threshold_y)[0]
    
    if len(spikes_x) == 0 and len(spikes_y) == 0:
        return 0.0
    
    if len(spikes_x) == 0 or len(spikes_y) == 0:
        return 1.0
    
    # Create convolved spike trains
    n = len(x)
    t = np.arange(n)
    
    # Convolve with exponential kernel
    fx = np.zeros(n)
    for spike in spikes_x:
        fx += np.exp(-np.abs(t - spike) / tau)
    
    fy = np.zeros(n)
    for spike in spikes_y:
        fy += np.exp(-np.abs(t - spike) / tau)
    
    # van Rossum distance = integral of squared difference
    vr_dist = np.sqrt(np.sum((fx - fy) ** 2))
    
    # Normalize by max possible distance
    max_dist = np.sqrt(np.sum(fx ** 2) + np.sum(fy ** 2))
    
    return vr_dist / max_dist if max_dist > 0 else 0.0


# ============================================================================
# Distance Method Registry
# ============================================================================

_DISTANCE_REGISTRY: Dict[str, Callable] = {
    'correlation': tsdist_cor,
    'cor': tsdist_cor,
    'ccf': tsdist_ccf,
    'cross_correlation': tsdist_ccf,
    'dtw': tsdist_dtw,
    'nmi': tsdist_nmi,
    'mutual_information': tsdist_nmi,
    'voi': tsdist_voi,
    'variation_of_information': tsdist_voi,
    'es': tsdist_es,
    'event_sync': tsdist_es,
    'mic': tsdist_mic,
    'maximal_information_coefficient': tsdist_mic,
    'vr': tsdist_vr,
    'van_rossum': tsdist_vr,
}


# ============================================================================
# Master Distance Function
# ============================================================================

def ts_dist(X: NDArray[np.float64], method: str = 'correlation', 
            n_jobs: int = 1, **kwargs) -> NDArray[np.float64]:
    """
    Calculate pairwise distance matrix between multiple time series.
    
    Parameters
    ----------
    X : array (n_series, n_timepoints)
        Multiple time series to compare
    method : str
        Distance function: 'correlation', 'ccf', 'dtw', 'nmi', 'es'
    n_jobs : int
        Number of parallel workers (-1 = all cores)
    **kwargs
        Distance-specific parameters
    
    Returns
    -------
    D : array (n_series, n_series)
        Distance matrix (symmetric, diagonal = 0)
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(10, 100)
    >>> D = ts_dist(X, method='correlation', n_jobs=-1)
    >>> D.shape
    (10, 10)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    n_series, n_points = X.shape
    
    # Get distance function
    dist_func = _DISTANCE_REGISTRY.get(method.lower())
    if dist_func is None:
        raise ValueError(f"Unknown distance method: {method}. "
                        f"Available: {list(_DISTANCE_REGISTRY.keys())}")
    
    logger.info(f"Computing {method} distance matrix for {n_series} series ({n_points} points each)")
    
    # Calculate distance matrix
    if n_jobs == 1:
        # Serial computation
        D = _compute_distance_matrix_serial(X, dist_func, **kwargs)
    else:
        # Parallel computation
        D = _compute_distance_matrix_parallel(X, dist_func, n_jobs, **kwargs)
    
    return D


def _compute_distance_matrix_serial(X: np.ndarray, dist_func: Callable, 
                                     **kwargs) -> np.ndarray:
    """Serial computation of distance matrix"""
    n_series = X.shape[0]
    D = np.zeros((n_series, n_series))
    
    for i in range(n_series):
        for j in range(i + 1, n_series):
            d = dist_func(X[i], X[j], **kwargs)
            D[i, j] = d
            D[j, i] = d
    
    return D


def _compute_distance_matrix_parallel(X: np.ndarray, dist_func: Callable, 
                                       n_jobs: int, **kwargs) -> np.ndarray:
    """Parallel computation of distance matrix"""
    n_series = X.shape[0]
    
    # Generate pairs
    pairs = [(i, j) for i in range(n_series) for j in range(i + 1, n_series)]
    
    # Compute distances in parallel
    # Use threading backend to avoid loky multiprocessing segfaults in test environments
    distances = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(dist_func)(X[i], X[j], **kwargs) for i, j in pairs
    )
    
    # Fill matrix
    D = np.zeros((n_series, n_series))
    for (i, j), d in zip(pairs, distances):
        D[i, j] = d
        D[j, i] = d
    
    return D


def ts_dist_part(X: NDArray[np.float64], start_idx: int, end_idx: int, 
                 method: str = 'correlation', **kwargs) -> NDArray[np.float64]:
    """
    Calculate partial distance matrix for HPC batch processing.
    
    Parameters
    ----------
    X : array (n_series, n_timepoints)
        All time series
    start_idx : int
        Start row index (inclusive)
    end_idx : int
        End row index (exclusive)
    method : str
        Distance function name
    **kwargs
        Distance-specific parameters
    
    Returns
    -------
    D_part : array (end_idx - start_idx, n_series)
        Partial distance matrix (rows start_idx:end_idx)
    
    Examples
    --------
    >>> # On cluster node 1
    >>> D_part = ts_dist_part(X, 0, 100, method='dtw')
    >>> np.save('D_part_0_100.npy', D_part)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    
    n_series = X.shape[0]
    
    if start_idx < 0 or end_idx > n_series or start_idx >= end_idx:
        raise ValueError(f"Invalid indices: start={start_idx}, end={end_idx}, n={n_series}")
    
    # Get distance function
    dist_func = _DISTANCE_REGISTRY.get(method.lower())
    if dist_func is None:
        raise ValueError(f"Unknown distance method: {method}")
    
    # Calculate partial matrix
    n_rows = end_idx - start_idx
    D_part = np.zeros((n_rows, n_series))
    
    for i in range(start_idx, end_idx):
        for j in range(n_series):
            if i == j:
                D_part[i - start_idx, j] = 0.0
            else:
                D_part[i - start_idx, j] = dist_func(X[i], X[j], **kwargs)
    
    logger.info(f"Computed partial distance matrix: rows {start_idx}:{end_idx}")
    
    return D_part

