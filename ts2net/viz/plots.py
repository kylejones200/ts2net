"""
Core plotting functions for ts2net.

All functions follow the contract:
- Accept raw series and derived arrays
- Return (fig, ax) tuple
- Use consistent Matplotlib styling
- Scale well to large datasets
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Consistent styling constants
FONT_FAMILY = 'DejaVu Sans'
FIG_WIDTH = 10
FIG_HEIGHT = 6
DPI = 100
GRID_ALPHA = 0.3
SPINE_COLOR = '#333333'


def _setup_style(ax, grid: bool = False):
    """Apply consistent styling to an axis."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(SPINE_COLOR)
    ax.spines['bottom'].set_color(SPINE_COLOR)
    
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    ax.tick_params(colors=SPINE_COLOR)


def plot_series_with_events(
    x: np.ndarray,
    events: Optional[np.ndarray] = None,
    window: Optional[Tuple[int, int]] = None,
    window_bounds: Optional[List[Tuple[int, int]]] = None,
    time_index: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Figure 1: Time series with change points and window boundaries.
    
    Shows the signal with detected change points as thin vertical lines
    and window edges as faint bands. Provides context for network results.
    
    Parameters
    ----------
    x : array (n,)
        Time series values
    events : array (m,), optional
        Indices of detected change points
    window : tuple (start, end), optional
        Single window boundaries to highlight
    window_bounds : list of tuples, optional
        Multiple window boundaries [(start, end), ...]
    time_index : array (n,), optional
        Time indices (default: 0, 1, 2, ...)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (10, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if time_index is None:
        time_index = np.arange(len(x))
    
    if figsize is None:
        figsize = (FIG_WIDTH, FIG_HEIGHT)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Plot main series
    ax.plot(time_index, x, 'k-', linewidth=1.5, alpha=0.8, label='Series')
    
    # Add window boundaries as faint bands
    if window_bounds:
        for start, end in window_bounds:
            ax.axvspan(start, end, alpha=0.1, color='blue', zorder=0)
    elif window:
        start, end = window
        ax.axvspan(start, end, alpha=0.1, color='blue', zorder=0)
    
    # Add change points as thin vertical lines
    if events is not None:
        for event_idx in events:
            if 0 <= event_idx < len(time_index):
                ax.axvline(time_index[event_idx], color='red', 
                          linewidth=1, alpha=0.6, linestyle='--', zorder=2)
    
    # Styling
    _setup_style(ax, grid=True)
    ax.set_xlabel('Time Index', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('Value', fontfamily=FONT_FAMILY, fontsize=11)
    
    if title:
        ax.set_title(title, fontfamily=FONT_FAMILY, fontsize=13, pad=10)
    
    plt.tight_layout()
    return fig, ax


def plot_degree_profile(
    degrees: np.ndarray,
    time_index: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Figure 2: Degree profile across time.
    
    Plots degree versus time index. This becomes a direct proxy for
    "local complexity" in the signal. Reads well and scales well.
    
    Parameters
    ----------
    degrees : array (n,)
        Degree sequence (one per node/time point)
    time_index : array (n,), optional
        Time indices (default: 0, 1, 2, ...)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (10, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if time_index is None:
        time_index = np.arange(len(degrees))
    
    if figsize is None:
        figsize = (FIG_WIDTH, FIG_HEIGHT)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Plot degree profile
    ax.plot(time_index, degrees, 'b-', linewidth=1.5, alpha=0.7)
    
    # Add mean line
    mean_deg = np.mean(degrees)
    ax.axhline(mean_deg, color='red', linewidth=1, alpha=0.5, 
              linestyle='--', label=f'Mean: {mean_deg:.2f}')
    
    # Styling
    _setup_style(ax, grid=True)
    ax.set_xlabel('Time Index', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('Degree', fontfamily=FONT_FAMILY, fontsize=11)
    
    if title:
        ax.set_title(title, fontfamily=FONT_FAMILY, fontsize=13, pad=10)
    
    # Annotation instead of legend
    ax.text(0.02, 0.98, f'Mean: {mean_deg:.2f}', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily=FONT_FAMILY,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig, ax


def plot_degree_ccdf(
    degrees: np.ndarray,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Figure 3: Degree distribution as CCDF (Complementary CDF).
    
    Plots the complementary CDF of degrees on log y scale. Works better
    than a histogram, stays stable across sample size, makes cross-zone
    comparison easy.
    
    Parameters
    ----------
    degrees : array (n,)
        Degree sequence
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (10, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if figsize is None:
        figsize = (FIG_WIDTH, FIG_HEIGHT)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Compute CCDF
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    n = len(degrees)
    cdf = np.cumsum(counts) / n
    ccdf = 1 - cdf
    
    # Plot on log scale
    ax.semilogy(unique_degrees, ccdf, 'o-', markersize=4, 
               linewidth=1.5, alpha=0.7, color='blue')
    
    # Styling
    _setup_style(ax, grid=True)
    ax.set_xlabel('Degree', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('CCDF (log scale)', fontfamily=FONT_FAMILY, fontsize=11)
    
    if title:
        ax.set_title(title, fontfamily=FONT_FAMILY, fontsize=13, pad=10)
    
    plt.tight_layout()
    return fig, ax


def plot_method_comparison(
    df_metrics: Union[dict, np.ndarray],
    methods: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Figure 4: Method comparison panel.
    
    Creates a small table graphic with three aligned dot plots:
    - Edge count
    - Average degree
    - Normalized density (edges / n)
    
    Uses one axis per metric. Zones/methods on y-axis.
    
    Parameters
    ----------
    df_metrics : dict or array
        Dictionary mapping method names to dicts with 'n_edges', 'avg_degree', 'density'
        OR array of dicts with 'method' key
    methods : list of str, optional
        Method names (if df_metrics is array)
    figsize : tuple, optional
        Figure size (default: (12, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    # Normalize input format
    if isinstance(df_metrics, dict):
        methods = list(df_metrics.keys())
        metrics_list = [df_metrics[m] for m in methods]
    else:
        if methods is None:
            methods = [m.get('method', f'Method_{i}') for i, m in enumerate(df_metrics)]
        metrics_list = df_metrics
    
    if figsize is None:
        figsize = (12, 6)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=DPI, sharey=True)
    fig.patch.set_facecolor('white')
    
    # Extract metrics
    n_edges = [m.get('n_edges', 0) for m in metrics_list]
    avg_degrees = [m.get('avg_degree', 0) for m in metrics_list]
    densities = [m.get('density', 0) for m in metrics_list]
    
    # If density not provided, compute from n_edges and n_nodes
    if all(d == 0 for d in densities):
        n_nodes = [m.get('n_nodes', 1) for m in metrics_list]
        densities = [e / (n * (n - 1) / 2) if n > 1 else 0 
                     for e, n in zip(n_edges, n_nodes)]
    
    y_pos = np.arange(len(methods))
    
    # Plot 1: Edge count
    axes[0].scatter(n_edges, y_pos, s=100, alpha=0.7, color='blue')
    axes[0].set_xlabel('Edge Count', fontfamily=FONT_FAMILY, fontsize=11)
    axes[0].set_ylabel('Method', fontfamily=FONT_FAMILY, fontsize=11)
    _setup_style(axes[0], grid=True)
    
    # Plot 2: Average degree
    axes[1].scatter(avg_degrees, y_pos, s=100, alpha=0.7, color='green')
    axes[1].set_xlabel('Avg Degree', fontfamily=FONT_FAMILY, fontsize=11)
    _setup_style(axes[1], grid=True)
    
    # Plot 3: Density
    axes[2].scatter(densities, y_pos, s=100, alpha=0.7, color='red')
    axes[2].set_xlabel('Density', fontfamily=FONT_FAMILY, fontsize=11)
    _setup_style(axes[2], grid=True)
    
    # Set y-axis labels
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(methods, fontfamily=FONT_FAMILY, fontsize=10)
    
    plt.tight_layout()
    return fig, axes


def plot_window_feature_map(
    df_window_features: Union[dict, np.ndarray],
    feature_names: Optional[List[str]] = None,
    time_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Figure 5: Window level feature map.
    
    Computes window stats (mean degree, degree variance, assortativity proxy,
    transition entropy) and plots as a heatmap with time on x and feature on y.
    Provides anomaly signatures.
    
    Parameters
    ----------
    df_window_features : dict or array
        Dictionary mapping feature names to arrays OR array of dicts
    feature_names : list of str, optional
        Feature names (if df_window_features is dict, uses keys)
    time_labels : list of str, optional
        Time window labels (default: 0, 1, 2, ...)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # Normalize input format
    if isinstance(df_window_features, dict):
        if feature_names is None:
            feature_names = list(df_window_features.keys())
        data_matrix = np.array([df_window_features[f] for f in feature_names])
    else:
        # Array of dicts - extract features
        if feature_names is None:
            # Infer from first dict
            feature_names = [k for k in df_window_features[0].keys() 
                           if k not in ['window_start', 'window_end']]
        data_matrix = np.array([[d.get(f, 0) for f in feature_names] 
                                for d in df_window_features])
    
    n_features, n_windows = data_matrix.shape
    
    if time_labels is None:
        time_labels = [f'Window {i}' for i in range(n_windows)]
    
    if figsize is None:
        figsize = (12, 8)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Normalize each feature for better visualization
    data_normalized = np.zeros_like(data_matrix)
    for i in range(n_features):
        col = data_matrix[i, :]
        if np.std(col) > 0:
            data_normalized[i, :] = (col - np.mean(col)) / np.std(col)
        else:
            data_normalized[i, :] = col
    
    # Create heatmap
    im = ax.imshow(data_normalized, aspect='auto', cmap='RdYlBu_r', 
                   interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(n_windows))
    ax.set_xticklabels(time_labels, rotation=45, ha='right', 
                      fontfamily=FONT_FAMILY, fontsize=9)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names, fontfamily=FONT_FAMILY, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Feature Value', fontfamily=FONT_FAMILY, fontsize=10)
    
    # Styling
    _setup_style(ax, grid=False)
    ax.set_xlabel('Time Window', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('Feature', fontfamily=FONT_FAMILY, fontsize=11)
    
    plt.tight_layout()
    return fig, ax


def plot_hvg_small(
    x: np.ndarray,
    edges: List[Tuple[int, int]],
    time_index: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Optional: Small n graph drawing for HVG.
    
    Uses fixed layout based on time index. Nodes at x = time, y = normalized value.
    Edges drawn as faint arcs or straight lines. Shows visibility logic.
    
    Parameters
    ----------
    x : array (n,)
        Time series values
    edges : list of tuples
        Edge list [(i, j), ...]
    time_index : array (n,), optional
        Time indices (default: 0, 1, 2, ...)
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (10, 6))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if len(x) > 200:
        raise ValueError("plot_hvg_small is for small series only (n <= 200)")
    
    if time_index is None:
        time_index = np.arange(len(x))
    
    if figsize is None:
        figsize = (FIG_WIDTH, FIG_HEIGHT)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Normalize values for y-position
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    
    # Plot edges as faint lines
    for i, j in edges:
        ax.plot([time_index[i], time_index[j]], 
               [x_norm[i], x_norm[j]], 
               'b-', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Plot nodes
    ax.scatter(time_index, x_norm, s=30, c='red', 
             alpha=0.8, zorder=2, edgecolors='black', linewidths=0.5)
    
    # Plot series line
    ax.plot(time_index, x_norm, 'k-', linewidth=1.5, alpha=0.5, zorder=0)
    
    # Styling
    _setup_style(ax, grid=True)
    ax.set_xlabel('Time Index', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('Normalized Value', fontfamily=FONT_FAMILY, fontsize=11)
    
    if title:
        ax.set_title(title, fontfamily=FONT_FAMILY, fontsize=13, pad=10)
    
    plt.tight_layout()
    return fig, ax


def plot_recurrence_matrix(
    recurrence_matrix: np.ndarray,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Optional: Recurrence plot style view for recurrence networks.
    
    Draws the recurrence matrix as an image for a short window.
    Users already understand this visual.
    
    Parameters
    ----------
    recurrence_matrix : array (n, n)
        Recurrence/adjacency matrix
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (default: (8, 8))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if figsize is None:
        figsize = (8, 8)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor('white')
    
    # Plot as image
    im = ax.imshow(recurrence_matrix, cmap='binary', origin='lower', 
                   interpolation='nearest', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Recurrence', fontfamily=FONT_FAMILY, fontsize=10)
    
    # Styling
    _setup_style(ax, grid=False)
    ax.set_xlabel('Time Index', fontfamily=FONT_FAMILY, fontsize=11)
    ax.set_ylabel('Time Index', fontfamily=FONT_FAMILY, fontsize=11)
    
    if title:
        ax.set_title(title, fontfamily=FONT_FAMILY, fontsize=13, pad=10)
    
    plt.tight_layout()
    return fig, ax
