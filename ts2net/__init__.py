"""
ts2net: Time Series to Networks

Clean API inspired by ts2vg, extended for multiple network methods.
"""

from .core.graph import Graph
from .api import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network
from .core import graph_summary

__version__ = "0.6.0"

__all__ = [
    'Graph',
    'HVG',
    'NVG',
    'RecurrenceNetwork',
    'TransitionNetwork',
    'build_network',
    'graph_summary',
]

# Optional Polars-based IO
try:
    from .io_polars import load_series_from_parquet_polars
    __all__.append('load_series_from_parquet_polars')
except ImportError:
    pass

# Visualization module
try:
    from .viz import (
        plot_series_with_events,
        plot_degree_profile,
        plot_degree_ccdf,
        plot_method_comparison,
        plot_window_feature_map,
        plot_hvg_small,
        plot_recurrence_matrix,
    )
    __all__.extend([
        'plot_series_with_events',
        'plot_degree_profile',
        'plot_degree_ccdf',
        'plot_method_comparison',
        'plot_window_feature_map',
        'plot_hvg_small',
        'plot_recurrence_matrix',
    ])
except ImportError:
    pass

# Columnar adapters
try:
    from .io_adapters import from_pandas, from_polars
    __all__.extend(['from_pandas', 'from_polars'])
except ImportError:
    pass

# Windowed graphs API
from .api_windows import build_windows
__all__.append('build_windows')

# Configuration and factory modules
from .config import PipelineConfig
from .factory import create_graph_builder, build_graph_from_config
__all__.extend(['PipelineConfig', 'create_graph_builder', 'build_graph_from_config'])

# BSTS decomposition and features (optional - requires statsmodels)
try:
    from .bsts import decompose, BSTSSpec, features
    __all__ = __all__ + ['decompose', 'BSTSSpec', 'features']
except ImportError:
    pass

# Temporal CNN embeddings (optional - requires torch)
try:
    from .temporal_cnn import temporal_cnn_embeddings
    __all__.append('temporal_cnn_embeddings')
except ImportError:
    pass
