"""
Null models and significance testing for network metrics.

This module provides surrogate data generation and statistical significance
testing for time series network analysis. It enables validation of network
properties against null distributions.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .stats import surrogate_phase, surrogate_circular, iaaft


SurrogateMethod = Literal["shuffle", "phase", "circular", "iaaft", "block_bootstrap"]


@dataclass
class NetworkSignificanceResult:
    """
    Results from network significance testing.
    
    Attributes
    ----------
    metric_name : str
        Name of the metric being tested
    observed_value : float
        Observed value of the metric
    null_mean : float
        Mean of the null distribution
    null_std : float
        Standard deviation of the null distribution
    z_score : float
        Z-score: (observed - mean) / std
    p_value : float
        Two-tailed p-value (approximate, based on normal distribution)
    n_surrogates : int
        Number of surrogates used
    surrogate_method : str
        Method used to generate surrogates
    significant : bool
        Whether the result is significant at alpha=0.05 (two-tailed)
    confidence_interval : tuple[float, float]
        95% confidence interval for the metric under the null
    """
    
    metric_name: str
    observed_value: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    n_surrogates: int
    surrogate_method: str
    significant: bool
    confidence_interval: tuple[float, float]
    alpha: float = 0.05
    
    def __str__(self) -> str:
        """String representation of the result."""
        sig_str = "significant" if self.significant else "not significant"
        return (
            f"{self.metric_name}: {self.observed_value:.4f} "
            f"(z={self.z_score:.3f}, p={self.p_value:.4f}, {sig_str})"
        )
    
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary summary of the result."""
        return {
            "metric": self.metric_name,
            "observed": self.observed_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "significant": self.significant,
            "n_surrogates": self.n_surrogates,
            "method": self.surrogate_method,
            "ci_95": self.confidence_interval,
        }


def generate_surrogate(
    x: NDArray[np.float64],
    method: SurrogateMethod = "shuffle",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> NDArray[np.float64]:
    """
    Generate a surrogate time series using the specified method.
    
    Parameters
    ----------
    x : array
        Original time series
    method : str, default "shuffle"
        Surrogate generation method:
        - "shuffle": Random permutation (destroys all structure)
        - "phase": Phase randomization (preserves power spectrum)
        - "circular": Circular shift (preserves all structure, shifts phase)
        - "iaaft": Iterative amplitude adjusted Fourier transform (preserves power spectrum and distribution)
        - "block_bootstrap": Block bootstrap (preserves local structure)
    rng : np.random.Generator, optional
        Random number generator
    **kwargs
        Additional arguments for specific methods:
        - block_bootstrap: block_size (int, default=10)
        - iaaft: iters (int, default=50)
    
    Returns
    -------
    surrogate : array
        Surrogate time series with same length as x
    
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> x_surr = generate_surrogate(x, method="phase")
    >>> assert len(x_surr) == len(x)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    x = np.asarray(x, dtype=np.float64)
    
    def _get_seed(rng):
        """Extract integer seed from rng."""
        return rng.integers(0, 2**31) if isinstance(rng, np.random.Generator) else (int(rng) if rng is not None else None)
    
    method_handlers = {
        "shuffle": lambda: rng.permutation(x),
        "phase": lambda: surrogate_phase(x, rng=_get_seed(rng)),
        "circular": lambda: np.roll(x, int(rng.integers(0, len(x)))),
        "iaaft": lambda: iaaft(x, iters=kwargs.get("iters", 50), rng=_get_seed(rng)),
        "block_bootstrap": lambda: _block_bootstrap(x, block_size=kwargs.get("block_size", 10), rng=rng),
    }
    
    handler = method_handlers.get(method)
    if handler is None:
        raise ValueError(
            f"Unknown surrogate method: {method}. "
            f"Choose from: {', '.join(method_handlers.keys())}"
        )
    
    return handler()


def _block_bootstrap(
    x: NDArray[np.float64],
    block_size: int = 10,
    rng: Optional[np.random.Generator] = None
) -> NDArray[np.float64]:
    """
    Generate surrogate using block bootstrap.
    
    Preserves local temporal structure by resampling blocks of consecutive points.
    
    Parameters
    ----------
    x : array
        Original time series
    block_size : int, default 10
        Size of blocks to resample
    rng : np.random.Generator, optional
        Random number generator
    
    Returns
    -------
    surrogate : array
        Block-bootstrapped time series
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x)
    n_blocks = (n + block_size - 1) // block_size  # Ceiling division
    result = np.zeros(n, dtype=x.dtype)
    
    for i in range(n_blocks):
        # Randomly select a starting point for this block
        start_idx = rng.integers(0, max(1, n - block_size + 1))
        end_idx = min(start_idx + block_size, n)
        
        # Copy block to result
        result_start = i * block_size
        result_end = min(result_start + block_size, n)
        block_len = result_end - result_start
        result[result_start:result_end] = x[start_idx:start_idx + block_len]
    
    return result


def compute_zscore(
    observed: float,
    null_values: NDArray[np.float64],
    ddof: int = 1
) -> tuple[float, float, float]:
    """
    Compute z-score from observed value and null distribution.
    
    Parameters
    ----------
    observed : float
        Observed metric value
    null_values : array
        Values from null distribution (surrogates)
    ddof : int, default 1
        Degrees of freedom for standard deviation calculation
    
    Returns
    -------
    z_score : float
        Z-score: (observed - mean) / std
    null_mean : float
        Mean of null distribution
    null_std : float
        Standard deviation of null distribution
    """
    null_mean = float(np.mean(null_values))
    null_std = float(np.std(null_values, ddof=ddof))
    
    # Avoid division by zero
    if null_std < 1e-12:
        null_std = 1e-12
    
    z_score = (observed - null_mean) / null_std
    
    return z_score, null_mean, null_std


def compute_network_metric_significance(
    x: NDArray[np.float64],
    metric_fn: Callable[[NDArray[np.float64]], float],
    method: SurrogateMethod = "shuffle",
    n_surrogates: int = 200,
    alpha: float = 0.05,
    metric_name: str = "metric",
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> NetworkSignificanceResult:
    """
    Test significance of a network metric against null distribution.
    
    Parameters
    ----------
    x : array
        Original time series
    metric_fn : callable
        Function that takes a time series and returns a scalar metric.
        Should build network and compute metric.
    method : str, default "shuffle"
        Surrogate generation method
    n_surrogates : int, default 200
        Number of surrogate series to generate
    alpha : float, default 0.05
        Significance level (two-tailed)
    metric_name : str, default "metric"
        Name of the metric (for reporting)
    rng : np.random.Generator, optional
        Random number generator
    **kwargs
        Additional arguments passed to generate_surrogate()
    
    Returns
    -------
    result : NetworkSignificanceResult
        Significance test result
    
    Examples
    --------
    >>> from ts2net import HVG
    >>> x = np.random.randn(100)
    >>> 
    >>> def compute_density(ts):
    ...     hvg = HVG()
    ...     hvg.build(ts)
    ...     stats = hvg.stats()
    ...     return stats['density']
    >>> 
    >>> result = compute_network_metric_significance(x, compute_density, method="phase", n_surrogates=100)
    >>> print(result)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute observed metric
    observed_value = float(metric_fn(x))
    
    # Generate surrogates and compute null distribution
    null_values = np.zeros(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        x_surr = generate_surrogate(x, method=method, rng=rng, **kwargs)
        null_values[i] = float(metric_fn(x_surr))
    
    # Compute z-score
    z_score, null_mean, null_std = compute_zscore(observed_value, null_values)
    
    # Compute p-value (two-tailed, approximate from normal distribution)
    from scipy import stats as sp_stats
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))
    
    # Determine significance
    significant = p_value < alpha
    
    # Compute confidence interval (95% by default, but use alpha)
    ci_level = 1 - alpha
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lower = null_mean - z_crit * null_std
    ci_upper = null_mean + z_crit * null_std
    confidence_interval = (ci_lower, ci_upper)
    
    return NetworkSignificanceResult(
        metric_name=metric_name,
        observed_value=observed_value,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        p_value=p_value,
        n_surrogates=n_surrogates,
        surrogate_method=method,
        significant=significant,
        confidence_interval=confidence_interval,
        alpha=alpha,
    )


def compute_multiple_metrics_significance(
    x: NDArray[np.float64],
    metric_fns: Dict[str, Callable[[NDArray[np.float64]], float]],
    method: SurrogateMethod = "shuffle",
    n_surrogates: int = 200,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> Dict[str, NetworkSignificanceResult]:
    """
    Test multiple network metrics simultaneously.
    
    More efficient than calling test_network_metric multiple times
    because surrogates are generated once and reused.
    
    Parameters
    ----------
    x : array
        Original time series
    metric_fns : dict
        Dictionary mapping metric names to functions
    method : str, default "shuffle"
        Surrogate generation method
    n_surrogates : int, default 200
        Number of surrogate series to generate
    alpha : float, default 0.05
        Significance level
    rng : np.random.Generator, optional
        Random number generator
    **kwargs
        Additional arguments passed to generate_surrogate()
    
    Returns
    -------
    results : dict
        Dictionary mapping metric names to NetworkSignificanceResult objects
    
    Examples
    --------
    >>> from ts2net import HVG
    >>> x = np.random.randn(100)
    >>> 
    >>> def compute_density(ts):
    ...     hvg = HVG()
    ...     hvg.build(ts)
    ...     return hvg.stats()['density']
    >>> 
    >>> def compute_clustering(ts):
    ...     hvg = HVG()
    ...     hvg.build(ts)
    ...     return hvg.stats()['avg_clustering']
    >>> 
    >>> metrics = {
    ...     "density": compute_density,
    ...     "clustering": compute_clustering,
    ... }
    >>> results = compute_multiple_metrics_significance(x, metrics, method="phase", n_surrogates=100)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute observed metrics
    observed_values = {}
    for name, fn in metric_fns.items():
        observed_values[name] = float(fn(x))
    
    # Generate surrogates once
    surrogates = []
    for _ in range(n_surrogates):
        x_surr = generate_surrogate(x, method=method, rng=rng, **kwargs)
        surrogates.append(x_surr)
    
    # Compute null distributions for each metric
    results = {}
    for name, metric_fn in metric_fns.items():
        null_values = np.array([float(metric_fn(x_surr)) for x_surr in surrogates])
        
        z_score, null_mean, null_std = compute_zscore(
            observed_values[name], null_values
        )
        
        from scipy import stats as sp_stats
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_score)))
        significant = p_value < alpha
        
        z_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci_lower = null_mean - z_crit * null_std
        ci_upper = null_mean + z_crit * null_std
        
        results[name] = NetworkSignificanceResult(
            metric_name=name,
            observed_value=observed_values[name],
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            p_value=p_value,
            n_surrogates=n_surrogates,
            surrogate_method=method,
            significant=significant,
            confidence_interval=(ci_lower, ci_upper),
            alpha=alpha,
        )
    
    return results

