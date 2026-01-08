"""
Network construction from time series data.

This module provides functions to construct networks from time series data
using various algorithms, including k-nearest neighbors, ε-nearest neighbors,
and weighted networks.
"""

from .core import (
    net_enn,
    net_knn,
    net_weighted,
    net_knn_approx,
    net_enn_approx,
)

from .metrics import (
    compute_clustering,
    compute_path_lengths,
    compute_modularity,
    network_metrics,
)

__all__ = [
    "net_enn",
    "net_knn",
    "net_weighted",
    "net_knn_approx",
    "net_enn_approx",
    "compute_clustering",
    "compute_path_lengths",
    "compute_modularity",
    "network_metrics",
]
