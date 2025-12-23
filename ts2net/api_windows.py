"""
Windowed graphs API for meter data and large time series.

Provides high-level API for building graph statistics per window,
storing only time series of stats (not full graphs).
"""

from __future__ import annotations

from typing import Optional, Dict, List, Union, Callable
import numpy as np
from numpy.typing import NDArray

from .multivariate.windows import ts_to_windows
from .factory import create_graph_builder, aggregate_stats
from .config import HVGConfig, NVGConfig, RecurrenceConfig, TransitionConfig


def build_windows(
    x: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: str = "hvg",
    output: str = "stats",
    aggregate: Optional[str] = None,
    **method_kwargs
) -> Dict[str, np.ndarray]:
    """
    Build graph statistics per window (memory efficient for large series).
    
    For meter data with millions of points, this computes graph stats per window
    and returns only the time series of stats, not full graphs.
    
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
        If None, returns full stats dict per window
    **method_kwargs
        Additional parameters for the network builder
    
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping stat names to arrays of length n_windows.
        If aggregate is None, returns nested dict with all stats.
        If aggregate is set, returns single aggregated stat per window.
    
    Examples
    --------
    >>> x = np.random.randn(10000)
    >>> # Compute window-level stats (memory efficient)
    >>> stats = build_windows(x, window=24, step=12, method='hvg', output='stats')
    >>> print(stats['n_nodes'].shape)  # (n_windows,)
    >>> print(stats['avg_degree'].shape)  # (n_windows,)
    
    >>> # With aggregation
    >>> avg_deg = build_windows(x, window=24, aggregate='mean', method='hvg')
    >>> # Returns single array: avg_degree per window
    """
    # Extract windows
    windows = ts_to_windows(x, width=window, by=step)
    n_windows = windows.shape[0]
    
    # Initialize result storage
    if aggregate:
        result = np.zeros(n_windows, dtype=np.float64)
    else:
        result = {
            'n_nodes': np.zeros(n_windows, dtype=np.int64),
            'n_edges': np.zeros(n_windows, dtype=np.int64),
            'avg_degree': np.zeros(n_windows, dtype=np.float64),
            'std_degree': np.zeros(n_windows, dtype=np.float64),
        }
    
    # Create config from method and kwargs using dispatch pattern
    def _create_hvg_config():
        return HVGConfig(
            enabled=True,
            output=output,
            weighted=method_kwargs.get('weighted', False),
            limit=method_kwargs.get('limit'),
            directed=method_kwargs.get('directed', False)
        )
    
    def _create_nvg_config():
        return NVGConfig(
            enabled=True,
            output=output,
            weighted=method_kwargs.get('weighted', False),
            limit=method_kwargs.get('limit', min(100, window)),
            max_edges=method_kwargs.get('max_edges'),
            max_edges_per_node=method_kwargs.get('max_edges_per_node'),
            max_memory_mb=method_kwargs.get('max_memory_mb')
        )
    
    def _create_recurrence_config():
        return RecurrenceConfig(
            enabled=True,
            output=output,
            m=method_kwargs.get('m', 3),
            rule='knn',
            k=method_kwargs.get('k', 5),
            tau=method_kwargs.get('tau', 1),
            epsilon=method_kwargs.get('epsilon', 0.1),
            metric=method_kwargs.get('metric', 'euclidean')
        )
    
    def _create_transition_config():
        return TransitionConfig(
            enabled=True,
            output=output,
            symbolizer=method_kwargs.get('symbolizer', 'ordinal'),
            order=method_kwargs.get('order', 3),
            n_states=method_kwargs.get('n_states')
        )
    
    config_map = {
        'hvg': _create_hvg_config,
        'nvg': _create_nvg_config,
        'recurrence': _create_recurrence_config,
        'transition': _create_transition_config,
    }
    
    config_factory = config_map.get(method.lower())
    if config_factory is None:
        raise ValueError(f"Unknown method: {method}. Must be one of {list(config_map.keys())}")
    
    config = config_factory()
    
    # Build network for each window
    for i, window_data in enumerate(windows):
        try:
            # Create builder using factory
            builder = create_graph_builder(method, config, n_points=len(window_data))
            builder.build(window_data)
            stats = builder.stats()
            
            if aggregate:
                # Store aggregated stat using dispatch
                result[i] = aggregate_stats(stats, aggregate)
            else:
                # Store all stats
                result['n_nodes'][i] = stats['n_nodes']
                result['n_edges'][i] = stats['n_edges']
                result['avg_degree'][i] = stats['avg_degree']
                result['std_degree'][i] = stats.get('std_degree', 0.0)
                
        except Exception as e:
            # Handle errors gracefully (e.g., constant windows)
            if aggregate:
                result[i] = np.nan
            else:
                result['n_nodes'][i] = 0
                result['n_edges'][i] = 0
                result['avg_degree'][i] = np.nan
                result['std_degree'][i] = np.nan
    
    return result
