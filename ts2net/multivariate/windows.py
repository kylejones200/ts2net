"""
Time series windowing utilities.

Provides functions to extract sliding windows from time series,
enabling proximity network construction (R ts2net compatibility).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def ts_to_windows(x: NDArray[np.float64], width: int, by: int = 1, 
                  start: int = 0, end: Optional[int] = None) -> NDArray[np.float64]:
    """
    Extract sliding windows from a time series.
    
    This function is equivalent to R ts2net's `ts_to_windows()` and enables
    proximity network construction where each window becomes a node.
    
    Parameters
    ----------
    x : array (n_points,)
        Input time series
    width : int
        Window width (number of time points per window)
    by : int
        Step size between consecutive windows
    start : int
        Starting index (0-based)
    end : int, optional
        Ending index (exclusive). If None, use len(x)
    
    Returns
    -------
    windows : array (n_windows, width)
        Matrix where each row is a window
    
    Examples
    --------
    >>> x = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> windows = ts_to_windows(x, width=10, by=1)
    >>> windows.shape
    (91, 10)
    
    >>> # Build proximity network from windows
    >>> from ts2net.multivariate import ts_dist, net_enn
    >>> D = ts_dist(windows, method='correlation', n_jobs=-1)
    >>> G, A = net_enn(D, percentile=20)
    
    Notes
    -----
    This implements the R ts2net approach for single time series:
    1. Extract sliding windows
    2. Treat each window as a time series
    3. Calculate pairwise distances
    4. Construct network (k-NN, ε-NN, etc.)
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1D array, got shape {x.shape}")
    
    n = len(x)
    
    if end is None:
        end = n
    
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    
    if by <= 0:
        raise ValueError(f"by must be positive, got {by}")
    
    if start < 0 or start >= n:
        raise ValueError(f"start must be in [0, {n-1}], got {start}")
    
    if end <= start or end > n:
        raise ValueError(f"end must be in ({start}, {n}], got {end}")
    
    if width > (end - start):
        raise ValueError(f"width ({width}) cannot exceed series length ({end - start})")
    
    # Calculate number of windows
    n_windows = (end - start - width) // by + 1
    
    if n_windows <= 0:
        raise ValueError(f"No windows possible with width={width}, by={by}, "
                        f"start={start}, end={end}")
    
    # Extract windows
    windows = np.zeros((n_windows, width))
    
    for i in range(n_windows):
        window_start = start + i * by
        window_end = window_start + width
        windows[i] = x[window_start:window_end]
    
    logger.info(f"Extracted {n_windows} windows of width {width} (step={by})")
    
    return windows


def ts_to_windows_list(X: List[np.ndarray], width: int, by: int = 1) -> NDArray[np.float64]:
    """
    Extract windows from multiple time series and concatenate.
    
    Parameters
    ----------
    X : list of arrays
        List of time series
    width : int
        Window width
    by : int
        Step size
    
    Returns
    -------
    windows : array (total_windows, width)
        All windows from all series
    
    Examples
    --------
    >>> series_list = [np.random.randn(100) for _ in range(5)]
    >>> windows = ts_to_windows_list(series_list, width=10, by=5)
    """
    all_windows = []
    
    for i, x in enumerate(X):
        try:
            windows = ts_to_windows(x, width=width, by=by)
            all_windows.append(windows)
        except ValueError as e:
            logger.warning(f"Skipping series {i}: {e}")
            continue
    
    if not all_windows:
        raise ValueError("No valid windows extracted from any series")
    
    return np.vstack(all_windows)


def ts_to_windows_labeled(x: NDArray[np.float64], width: int, by: int = 1) -> tuple:
    """
    Extract windows with temporal labels.
    
    Returns both windows and their starting indices for temporal analysis.
    
    Parameters
    ----------
    x : array
        Input time series
    width : int
        Window width
    by : int
        Step size
    
    Returns
    -------
    windows : array (n_windows, width)
        Window matrix
    indices : array (n_windows,)
        Starting index of each window
    
    Examples
    --------
    >>> x = np.arange(100)
    >>> windows, indices = ts_to_windows_labeled(x, width=10, by=5)
    >>> indices[:5]
    array([ 0,  5, 10, 15, 20])
    """
    windows = ts_to_windows(x, width=width, by=by)
    n_windows = windows.shape[0]
    indices = np.arange(n_windows) * by
    
    return windows, indices


def ts_window_stats(windows: NDArray[np.float64]) -> dict:
    """
    Calculate statistics for each window.
    
    Useful for feature extraction before network construction.
    
    Parameters
    ----------
    windows : array (n_windows, width)
        Window matrix
    
    Returns
    -------
    stats : dict
        Dictionary with arrays of statistics:
        - mean, std, min, max, median
        - skewness, kurtosis
        - trend (linear regression slope)
    
    Examples
    --------
    >>> windows = ts_to_windows(x, width=10, by=1)
    >>> stats = ts_window_stats(windows)
    >>> print(stats['mean'].shape)
    (n_windows,)
    """
    from scipy import stats as sp_stats
    
    n_windows = windows.shape[0]
    
    result = {
        'mean': np.mean(windows, axis=1),
        'std': np.std(windows, axis=1),
        'min': np.min(windows, axis=1),
        'max': np.max(windows, axis=1),
        'median': np.median(windows, axis=1),
        'skewness': sp_stats.skew(windows, axis=1),
        'kurtosis': sp_stats.kurtosis(windows, axis=1),
    }
    
    # Calculate trend (linear regression slope for each window)
    trends = np.zeros(n_windows)
    x_vals = np.arange(windows.shape[1])
    
    for i in range(n_windows):
        slope, _ = np.polyfit(x_vals, windows[i], 1)
        trends[i] = slope
    
    result['trend'] = trends
    
    return result

