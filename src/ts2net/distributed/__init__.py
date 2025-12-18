"""
Distributed computation utilities for large-scale time series analysis.

This module provides tools for parallel and distributed computation of
time series distances and network construction.
"""

from .core import DistJobConfig, ts_dist_part_file, ts_dist_merge_parts

__all__ = [
    "DistJobConfig",
    "ts_dist_part_file",
    "ts_dist_merge_parts",
]
