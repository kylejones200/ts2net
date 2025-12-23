"""
Graph drawing functions for TSGraph objects.

Provides unified rendering for time-series-derived graphs with
consistent styling and node coloring options.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .graph import TSGraph


def draw_tsgraph(
    tsgraph: TSGraph,
    *,
    ax=None,
    node_size: float = 10.0,
    edge_alpha: float = 0.15,
    node_alpha: float = 0.9,
    color_by: Literal["time", "community", "degree", "none"] = "time",
    cmap: str = "viridis",
    show: bool = True,
    layout: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Draw graph with thin edges and colored nodes.

    Expects tsgraph.pos. Falls back to a layout if pos is None.

    Parameters
    ----------
    tsgraph : TSGraph
        Graph container with graph, pos, and meta
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (creates new figure if None)
    node_size : float, default 10.0
        Size of nodes in points
    edge_alpha : float, default 0.15
        Transparency of edges (0-1)
    node_alpha : float, default 0.9
        Transparency of nodes (0-1)
    color_by : str, default "time"
        Node coloring scheme:
        - "time": Color by time index (uses node attribute 't')
        - "degree": Color by node degree
        - "community": Color by community (requires community detection)
        - "none": Single color for all nodes
    cmap : str, default "viridis"
        Colormap name for node colors
    show : bool, default True
        If True, call plt.show()
    layout : str, optional
        Layout algorithm if pos is None (e.g., "spring", "kamada_kawai")
        Defaults to "spring" for undirected, "circular" for directed

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    G = tsgraph.graph
    pos = tsgraph.pos
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        fig.patch.set_facecolor('white')
    else:
        fig = ax.figure
    
    # Get or compute node positions
    if pos is None:
        pos = _compute_layout(G, layout=layout)
    
    # Get node colors based on color_by
    node_colors = _get_node_colors(G, color_by=color_by, cmap=cmap)
    
    # Draw edges first (so nodes appear on top)
    _draw_edges(G, pos, ax, alpha=edge_alpha)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_size,
        node_color=node_colors,
        alpha=node_alpha,
        cmap=cmap if color_by != "none" else None,
    )
    
    # Remove axes ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def _compute_layout(G: nx.Graph, layout: Optional[str] = None) -> dict:
    """Compute node positions using a layout algorithm."""
    if layout is None:
        # Default: spring for undirected, circular for directed
        if isinstance(G, nx.DiGraph):
            layout = "circular"
        else:
            layout = "spring"
    
    if layout == "spring":
        return nx.spring_layout(G, k=1, iterations=50)
    elif layout == "circular":
        return nx.circular_layout(G)
    elif layout == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except:
            return nx.spring_layout(G, k=1, iterations=50)
    elif layout == "spectral":
        try:
            return nx.spectral_layout(G)
        except:
            return nx.spring_layout(G, k=1, iterations=50)
    else:
        return nx.spring_layout(G, k=1, iterations=50)


def _get_node_colors(
    G: nx.Graph, 
    color_by: Literal["time", "community", "degree", "none"],
    cmap: str = "viridis"
) -> list:
    """Get node colors based on coloring scheme."""
    n = G.number_of_nodes()
    
    if color_by == "none":
        return ['#1f77b4'] * n  # Single blue color
    
    if color_by == "time":
        # Use 't' attribute if available, otherwise node index
        times = [G.nodes[i].get('t', i) for i in range(n)]
        return times
    
    if color_by == "degree":
        degrees = [G.degree(i) for i in range(n)]
        return degrees
    
    if color_by == "community":
        # Try to detect communities
        try:
            if isinstance(G, nx.DiGraph):
                # Convert to undirected for community detection
                G_undir = G.to_undirected()
            else:
                G_undir = G
            
            communities = nx.community.greedy_modularity_communities(G_undir)
            community_map = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = comm_id
            
            colors = [community_map.get(i, 0) for i in range(n)]
            return colors
        except:
            # Fallback to degree if community detection fails
            degrees = [G.degree(i) for i in range(n)]
            return degrees
    
    # Default: single color
    return ['#1f77b4'] * n


def _draw_edges(G: nx.Graph, pos: dict, ax, alpha: float = 0.15):
    """Draw edges with thin lines and low alpha."""
    if isinstance(G, nx.DiGraph):
        # Draw directed edges as arrows
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            alpha=alpha,
            width=0.5,
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            edge_color='gray',
        )
    else:
        # Draw undirected edges as simple lines
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            alpha=alpha,
            width=0.5,
            edge_color='gray',
        )
