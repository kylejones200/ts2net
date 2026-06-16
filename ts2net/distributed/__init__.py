"""
Distributed computation utilities for large-scale time series analysis.

.. warning::
    **Experimental API** — This module is not part of the stable public API.
    Interfaces may change or be removed without notice. Prefer
    ``ts2net.multivariate.ts_dist`` with ``n_jobs=-1`` for parallel distance
    computation in production workflows.
"""

import warnings

warnings.warn(
    "ts2net.distributed is experimental and may change without notice.",
    FutureWarning,
    stacklevel=2,
)

from .core import DistJobConfig, ts_dist_part_file, ts_dist_merge_parts

__all__ = [
    "DistJobConfig",
    "ts_dist_part_file",
    "ts_dist_merge_parts",
]
