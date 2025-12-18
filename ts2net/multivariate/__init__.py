"""
Multivariate time series to network construction.

This module provides tools to construct networks from multiple time series,
where nodes represent time series and edges represent similarities/associations.

Examples
--------
>>> import numpy as np
>>> from ts2net.multivariate import ts_dist, net_knn
>>> X = np.random.randn(10, 100)  # 10 time series, 100 points each
>>> D = ts_dist(X, method='correlation', n_jobs=-1)
>>> G, A = net_knn(D, k=3)
>>> print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
"""

from .distances import ts_dist, ts_dist_part
from .builders import net_knn, net_enn, net_weighted, net_knn_approx, net_enn_approx
from .windows import ts_to_windows, ts_to_windows_list, ts_to_windows_labeled, ts_window_stats

__all__ = [
    # Distance calculation
    'ts_dist',
    'ts_dist_part',
    # Network construction
    'net_knn',
    'net_enn',
    'net_weighted',
    'net_knn_approx',
    'net_enn_approx',
    # Windowing utilities
    'ts_to_windows',
    'ts_to_windows_list',
    'ts_to_windows_labeled',
    'ts_window_stats',
]

