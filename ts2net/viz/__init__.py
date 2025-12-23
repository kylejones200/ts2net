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

from .graph import TSGraph, build_visibility_graph
from .draw import draw_tsgraph

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
    'draw_tsgraph',
]
