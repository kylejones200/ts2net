"""
Distributed computation utilities for large-scale time series analysis.

Prefer :func:`ts2net.scale.ts_dist_distributed` and
:func:`ts2net.scale.build_windows_distributed` for Dask/Ray workflows.
The CSV shard helpers below remain experimental.
"""

import warnings

warnings.warn(
    "ts2net.distributed CSV shard API is experimental; prefer ts2net.scale.distributed.",
    FutureWarning,
    stacklevel=2,
)

from ..scale.distributed import (
    build_windows_distributed,
    dask_available,
    parallel_map,
    ray_available,
    ts_dist_distributed,
)
from .core import DistJobConfig, ts_dist_merge_parts, ts_dist_part_file

__all__ = [
    "DistJobConfig",
    "ts_dist_part_file",
    "ts_dist_merge_parts",
    "parallel_map",
    "ts_dist_distributed",
    "build_windows_distributed",
    "dask_available",
    "ray_available",
]
