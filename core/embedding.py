"""
Time series embedding utilities including FNN-based dimension selection.

Implements False Nearest Neighbors (FNN) algorithm for automatic
embedding dimension estimation.
"""
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _embed_vectors_numba(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Fast time-delay embedding with Numba."""
    n = len(x)
    k = (m - 1) * tau
    n_vectors = n - k
    embedded = np.empty((n_vectors, m), dtype=x.dtype)
    
    for i in range(n_vectors):
        for j in range(m):
            embedded[i, j] = x[i + j * tau]
    
    return embedded


def embed_timeseries(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Create time-delay embedding of time series.
    
    Args:
        x: 1D time series
        m: Embedding dimension
        tau: Time delay
        
    Returns:
        Embedded vectors of shape (n_vectors, m)
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    n = len(x)
    k = (m - 1) * tau
    
    if k >= n:
        raise ValueError(
            f"Embedding dimension too large: n={n} < required={k+1} "
            f"for m={m}, tau={tau}"
        )
    
    if HAS_NUMBA:
        return _embed_vectors_numba(x, m, tau)
    
    # Fallback
    n_vectors = n - k
    return np.array([x[i:i+k+1:tau] for i in range(n_vectors)])


def false_nearest_neighbors(
    x: np.ndarray,
    tau: int = 1,
    max_dim: int = 10,
    rtol: float = 10.0,
    atol: float = 2.0
) -> Tuple[int, np.ndarray]:
    """
    Estimate optimal embedding dimension using False Nearest Neighbors.
    
    Algorithm from Kennel, Brown & Abarbanel (1992).
    
    Args:
        x: 1D time series
        tau: Time delay
        max_dim: Maximum dimension to test
        rtol: Relative tolerance threshold (default: 10)
        atol: Absolute tolerance threshold (default: 2)
        
    Returns:
        Tuple of (optimal_dim, fnn_percentages)
        
    References:
        Kennel, M. B., Brown, R., & Abarbanel, H. D. (1992).
        Determining embedding dimension for phase-space reconstruction
        using a geometrical construction. Physical Review A, 45(6), 3403.
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    n = len(x)
    
    if n < 100:
        logger.warning(f"Short series (n={n}) may give unreliable FNN results")
    
    fnn_percentages = []
    
    for m in range(1, max_dim + 1):
        # Embed in dimension m
        X_m = embed_timeseries(x, m, tau)
        n_m = len(X_m)
        
        if n_m < 10:
            logger.warning(f"Too few vectors at m={m}, stopping")
            break
        
        # Embed in dimension m+1
        if (m * tau) >= n - 1:
            break
        X_m1 = embed_timeseries(x, m + 1, tau)
        
        # Find nearest neighbors in m dimensions
        n_false = 0
        n_total = 0
        
        # Only iterate over points that exist in both embeddings
        n_compare = min(len(X_m), len(X_m1))
        
        for i in range(n_compare):
            # Compute distances in m dimensions
            dists_m = np.sum((X_m[:n_compare] - X_m[i])**2, axis=1)
            
            # Exclude self and temporally close neighbors
            dists_m[max(0, i-1):min(n_compare, i+2)] = np.inf
            
            # Find nearest neighbor
            nn_idx = np.argmin(dists_m)
            
            if np.isinf(dists_m[nn_idx]) or nn_idx >= n_compare:
                continue
            
            # Distance in m dimensions
            R_m = np.sqrt(dists_m[nn_idx])
            
            if R_m < 1e-10:
                continue
            
            # Distance in m+1 dimensions  
            R_m1 = np.linalg.norm(X_m1[i] - X_m1[nn_idx])
            
            # Check if neighbor is false
            # Criterion 1: Relative increase
            rel_increase = np.abs(R_m1 - R_m) / R_m
            
            # Criterion 2: Escape from attractor (compare to data std)
            Ra = np.std(x)
            abs_check = R_m1 / Ra
            
            if rel_increase > rtol or abs_check > atol:
                n_false += 1
            
            n_total += 1
        
        # Percentage of false nearest neighbors
        fnn_pct = (n_false / n_total * 100) if n_total > 0 else 100.0
        fnn_percentages.append(fnn_pct)
        
        logger.debug(f"FNN: m={m}, fnn={fnn_pct:.1f}%")
        
        # Stop if FNN percentage is low enough
        if fnn_pct < 1.0 and m >= 2:
            logger.info(f"FNN selected m={m} (FNN={fnn_pct:.1f}%)")
            return m, np.array(fnn_percentages)
    
    # Default: use dimension with minimum FNN
    if fnn_percentages:
        optimal_m = int(np.argmin(fnn_percentages) + 1)
        logger.info(f"FNN selected m={optimal_m} (minimum FNN)")
        return optimal_m, np.array(fnn_percentages)
    
    # Fallback
    logger.warning("FNN failed, defaulting to m=3")
    return 3, np.array(fnn_percentages)


def estimate_time_delay(
    x: np.ndarray,
    method: str = "autocorr",
    max_lag: int = 50
) -> int:
    """
    Estimate optimal time delay for embedding.
    
    Args:
        x: 1D time series
        method: "autocorr" (first zero crossing) or "mutual_info"
        max_lag: Maximum lag to consider
        
    Returns:
        Estimated time delay
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    
    if method == "autocorr":
        # First zero crossing of autocorrelation
        acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        
        # Find first zero crossing
        for lag in range(1, min(max_lag, len(acf))):
            if acf[lag] < 0:
                logger.info(f"Time delay selected: tau={lag} (ACF zero crossing)")
                return lag
        
        # Default: 1/10 of series length
        tau = max(1, len(x) // 10)
        logger.warning(f"No zero crossing found, using tau={tau}")
        return tau
    
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def auto_embed_params(
    x: np.ndarray,
    max_dim: int = 10
) -> Tuple[int, int]:
    """
    Automatically determine both embedding dimension and time delay.
    
    Args:
        x: 1D time series
        max_dim: Maximum dimension to test
        
    Returns:
        Tuple of (m, tau)
    """
    # First estimate time delay
    tau = estimate_time_delay(x)
    
    # Then estimate dimension
    m, _ = false_nearest_neighbors(x, tau=tau, max_dim=max_dim)
    
    logger.info(f"Auto-embedding selected: m={m}, tau={tau}")
    
    return m, tau

