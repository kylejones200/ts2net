"""
Parallel processing utilities for time series to network conversion.

This module provides functions for batch processing time series data in parallel.
"""

from typing import List, Tuple, Dict, Any, Sequence
import multiprocessing as mp
import numpy as np

# Third-party imports
try:
    import networkx as nx
except ImportError:
    nx = None

# Local imports
from .visibility import HVG, NVG
from .recurrence import RecurrenceNetwork
from .transition import TransitionNetwork


def _run_single(
    series: np.ndarray, builder: str, kwargs: dict
) -> Tuple[nx.Graph, np.ndarray]:
    """Convert a single time series to a network."""
    
    _BUILDERS = {
        "HVG": lambda s, kw: HVG(**kw).fit_transform(s),
        "NVG": lambda s, kw: NVG(**kw).fit_transform(s),
        "RN": lambda s, kw: RecurrenceNetwork(**kw).fit_transform(s),
        "TN": lambda s, kw: TransitionNetwork(**kw).fit_transform(s),
    }
    
    builder_fn = _BUILDERS.get(builder)
    if builder_fn is None:
        raise ValueError(f"Unknown builder: {builder}")
    
    return builder_fn(series, kwargs)


def batch_transform(
    X: Sequence[np.ndarray], builder: str, n_jobs: int = -1, **kwargs
) -> List[Tuple[nx.Graph, np.ndarray]]:
    """Convert multiple time series to networks in parallel.

    Parameters:
    -----------
    X : sequence of array-like
        List of time series to convert.
    builder : str
        Type of network to build. One of: "HVG", "NVG", "RN", "TN".
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all available cores.
    **kwargs : dict
        Additional keyword arguments to pass to the network builder.

    Returns:
    --------
    results : list of tuples
        List of (graph, adjacency_matrix) tuples, one for each input time series.
    """
    # Convert input to numpy arrays and ensure 1D
    arrs = [np.asarray(a, dtype=float).ravel() for a in X]

    # Determine number of processes to use
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)
    n_jobs = min(n_jobs, len(arrs))

    if n_jobs == 1:
        # Run in serial for small inputs or when n_jobs=1
        results = [_run_single(a, builder, kwargs) for a in arrs]
    else:
        # Run in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(_run_single, [(a, builder, kwargs) for a in arrs])

    return results
