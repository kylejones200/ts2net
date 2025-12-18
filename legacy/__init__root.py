"""
ts2net: Time series to network analysis tools.

This package provides tools for converting time series data into network representations
and analyzing the resulting networks. It includes implementations of various algorithms
for time series network construction, including visibility graphs, recurrence networks,
and transition networks.

Key features:
- Horizontal and Natural Visibility Graphs (HVG, NVG)
- Recurrence Networks (RN)
- Transition Networks (TN)
- Network analysis and visualization
- Distance metrics for time series
- Event detection and synchronization

Performance Notes:
- The package uses Rust implementations by default for better performance
- Python fallbacks are available if Rust extensions are not installed
- To ensure optimal performance, install the Rust extensions with: pip install -e .
"""

# Core network construction
try:
    from .core import (
        HVG,
        NVG,
        RecurrenceNetwork,
        TransitionNetwork,
        graph_summary,
        batch_transform,
    )

    __all__ = [
        "HVG",
        "NVG",
        "RecurrenceNetwork",
        "TransitionNetwork",
        "graph_summary",
        "batch_transform",
    ]
except ImportError as e:
    import warnings

    warnings.warn(f"Could not import core modules: {e}")
    __all__ = []

# Distance metrics
from .distances import (
    tsdist_cor,
    tsdist_ccf,
    tsdist_dtw,
    tsdist_nmi,
    tsdist_voi,
    tsdist_mic,
    tsdist_vr,
    dist_percentile,
    dist_matrix_normalize,
)

# Network construction
from .networks import net_enn, net_knn, net_weighted, net_knn_approx, net_enn_approx

# Event detection and synchronization
from .events import (
    events_from_ts,
    tssim_event_sync,
    random_ets,
    event_sync_full,
    EventSyncResult,
)

# Distributed computation
from .distributed import DistJobConfig, ts_dist_part_file, ts_dist_merge_parts

# Update __all__ with new exports
__all__.extend(
    [
        # Distance metrics
        "tsdist_cor",
        "tsdist_ccf",
        "tsdist_dtw",
        "tsdist_nmi",
        "tsdist_voi",
        "tsdist_mic",
        "tsdist_vr",
        "dist_percentile",
        "dist_matrix_normalize",
        # Network construction
        "net_enn",
        "net_knn",
        "net_weighted",
        "net_knn_approx",
        "net_enn_approx",
        # Event detection
        "events_from_ts",
        "tssim_event_sync",
        "random_ets",
        "event_sync_full",
        "EventSyncResult",
        # Distributed computation
        "DistJobConfig",
        "ts_dist_part_file",
        "ts_dist_merge_parts",
    ]
)

# Version
__version__ = "0.2.0"
