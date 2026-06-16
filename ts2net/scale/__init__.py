"""
Scale and performance utilities (horizon 0.6 / v0.6).
"""

from .approximate import (
    approximate_knn_network,
    approximate_knn_panel,
    has_pynndescent,
    should_use_approximate,
)
from .contracts import (
    PerformanceContract,
    get_performance_contract,
    list_performance_contracts,
)
from .incremental import AppendResult, IncrementalHVG
from .sparse import edges_to_csr, to_sparse_csr
from .streaming import (
    build_windows_streaming,
    estimate_window_job_memory_mb,
    iter_parquet_value_chunks,
    iter_series_chunks,
    iter_windows,
    stream_chunk_stats,
)

__all__ = [
    "PerformanceContract",
    "get_performance_contract",
    "list_performance_contracts",
    "IncrementalHVG",
    "AppendResult",
    "approximate_knn_network",
    "approximate_knn_panel",
    "has_pynndescent",
    "should_use_approximate",
    "to_sparse_csr",
    "edges_to_csr",
    "iter_windows",
    "iter_series_chunks",
    "iter_parquet_value_chunks",
    "build_windows_streaming",
    "stream_chunk_stats",
    "estimate_window_job_memory_mb",
]
