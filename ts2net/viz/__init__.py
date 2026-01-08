"""
Visualization module for ts2net.

Clean, scalable plotting functions for time series network analysis.
All functions return (fig, ax) for further customization.
"""

from .plots import (
    plot_series_with_events,
    plot_degree_profile,
    plot_degree_ccdf,
    plot_method_comparison,
    plot_window_feature_map,
    plot_hvg_small,
    plot_recurrence_matrix,
)

from .graph import (
    TSGraph,
    build_visibility_graph,
    build_recurrence_graph,
    build_ordinal_partition_graph,
    optimal_lag,
    optimal_dim,
)
from .draw import draw_tsgraph

# Optional Plotly visualizations
try:
    from .plotly_viz import plot_timeseries_network, plot_windowed_networks
    __all__ = [
        'plot_series_with_events',
        'plot_degree_profile',
        'plot_degree_ccdf',
        'plot_method_comparison',
        'plot_window_feature_map',
        'plot_hvg_small',
        'plot_recurrence_matrix',
        'TSGraph',
        'build_visibility_graph',
        'build_recurrence_graph',
        'build_ordinal_partition_graph',
        'optimal_lag',
        'optimal_dim',
        'draw_tsgraph',
        'plot_timeseries_network',
        'plot_windowed_networks',
    ]
except ImportError:
    __all__ = [
        'plot_series_with_events',
        'plot_degree_profile',
        'plot_degree_ccdf',
        'plot_method_comparison',
        'plot_window_feature_map',
        'plot_hvg_small',
        'plot_recurrence_matrix',
        'TSGraph',
        'build_visibility_graph',
        'build_recurrence_graph',
        'build_ordinal_partition_graph',
        'optimal_lag',
        'optimal_dim',
        'draw_tsgraph',
    ]
