"""
Interactive Plotly-based visualizations for time series networks.

Provides functions to create interactive network visualizations that show
how networks evolve over time using Plotly sliders.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union
import numpy as np
import networkx as nx

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plotly_plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    plotly_plot = None


def plot_timeseries_network(
    graphs: List[nx.Graph],
    timestamps: List[Any],
    pos: Optional[Dict[int, np.ndarray]] = None,
    node_colors: Optional[Union[str, List[str], Dict[int, str]]] = None,
    title: str = "Time Series Network Evolution",
    show: bool = True,
    filename: Optional[str] = None,
) -> go.Figure:
    """
    Create an interactive Plotly visualization showing network evolution over time.
    
    Parameters
    ----------
    graphs : list of nx.Graph
        List of NetworkX graphs, one for each time step
    timestamps : list
        List of timestamps/labels for each time step (e.g., dates, indices)
    pos : dict, optional
        Node positions dictionary. If None, computes spring layout from first graph
    node_colors : str, list, or dict, optional
        Node coloring scheme:
        - "degree": Color by node degree
        - "time": Color by time index
        - list: List of colors for each node
        - dict: Mapping of node -> color
    title : str, default "Time Series Network Evolution"
        Plot title
    show : bool, default True
        If True, display the plot
    filename : str, optional
        If provided, save the plot to this HTML file
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure with time slider
    
    Examples
    --------
    >>> from ts2net.viz.plotly_viz import plot_timeseries_network
    >>> import networkx as nx
    >>> 
    >>> # Create example graphs for different time steps
    >>> graphs = [nx.erdos_renyi_graph(20, 0.3) for _ in range(5)]
    >>> timestamps = [f"Step {i}" for i in range(5)]
    >>> 
    >>> fig = plot_timeseries_network(graphs, timestamps)
    >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )
    
    if len(graphs) != len(timestamps):
        raise ValueError(f"Number of graphs ({len(graphs)}) must match number of timestamps ({len(timestamps)})")
    
    if len(graphs) == 0:
        raise ValueError("At least one graph is required")
    
    # Compute positions if not provided
    if pos is None:
        pos = nx.spring_layout(graphs[0], k=1, iterations=50)
    
    # Create figure
    fig = go.Figure()
    
    # Build traces for each time step
    steps = []
    all_traces = []
    
    for t_idx, (G, timestamp) in enumerate(zip(graphs, timestamps)):
        # Get node positions for nodes that exist in this graph
        node_x = []
        node_y = []
        node_ids = []
        node_text = []
        
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_ids.append(node)
                node_text.append(f"Node {node}")
        
        # Get edge coordinates
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        # Get node colors
        colors = _get_node_colors_plotly(G, node_colors, node_ids)
        
        # Add edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            showlegend=False,
            name=f"Edges-{t_idx}",
        )
        
        # Add node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=10,
                color=colors,
                colorscale='Viridis',
                showscale=False,
                line=dict(width=0.5, color='white'),
            ),
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            name=f"Nodes-{t_idx}",
        )
        
        # Add traces to figure (initially hidden)
        fig.add_trace(edge_trace)
        fig.add_trace(node_trace)
        
        all_traces.append((len(fig.data) - 2, len(fig.data) - 1))
        
        # Create step for slider
        visibility = [False] * len(fig.data)
        visibility[len(fig.data) - 2] = True  # Edge trace
        visibility[len(fig.data) - 1] = True  # Node trace
        
        step = dict(
            method="update",
            label=str(timestamp),
            args=[
                {"visible": visibility},
                {"title": f"{title} - {timestamp}"}
            ],
        )
        steps.append(step)
    
    # Make first time step visible
    if len(all_traces) > 0:
        first_edge_idx, first_node_idx = all_traces[0]
        visibility = [False] * len(fig.data)
        visibility[first_edge_idx] = True
        visibility[first_node_idx] = True
        fig.data[first_edge_idx].visible = True
        fig.data[first_node_idx].visible = True
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Use the slider to navigate through time",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.005,
                y=-0.002,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12, color="gray"),
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Time: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )
    
    if filename:
        plotly_plot(fig, filename=filename, auto_open=False)
    
    if show:
        fig.show()
    
    return fig


def _get_node_colors_plotly(
    G: nx.Graph,
    node_colors: Optional[Union[str, List[str], Dict[int, str]]],
    node_ids: List[int]
) -> List[Any]:
    """Get node colors for Plotly visualization."""
    n = len(node_ids)
    
    if node_colors is None:
        # Default: color by degree
        return [G.degree(node_id) for node_id in node_ids]
    
    if isinstance(node_colors, str):
        if node_colors == "degree":
            return [G.degree(node_id) for node_id in node_ids]
        elif node_colors == "time":
            return node_ids  # Use node ID as time index
        else:
            # Unknown string, default to degree
            return [G.degree(node_id) for node_id in node_ids]
    
    if isinstance(node_colors, list):
        if len(node_colors) == n:
            return node_colors
        else:
            # Pad or truncate
            return node_colors[:n] + [node_colors[-1] if node_colors else 0] * (n - len(node_colors))
    
    if isinstance(node_colors, dict):
        return [node_colors.get(node_id, 0) for node_id in node_ids]
    
    # Default: degree
    return [G.degree(node_id) for node_id in node_ids]


def plot_windowed_networks(
    x: np.ndarray,
    window: int,
    step: int = 1,
    method: str = "hvg",
    pos: Optional[Dict[int, np.ndarray]] = None,
    **method_kwargs
) -> go.Figure:
    """
    Create interactive visualization of networks built from sliding windows.
    
    Parameters
    ----------
    x : array (n_points,)
        Input time series
    window : int
        Window width (number of time points per window)
    step : int, default 1
        Step size between consecutive windows
    method : str, default "hvg"
        Network method: 'hvg', 'nvg', 'recurrence', 'transition'
    pos : dict, optional
        Node positions. If None, computes from first window
    **method_kwargs
        Additional parameters for the network builder
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    
    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.viz.plotly_viz import plot_windowed_networks
    >>> 
    >>> x = np.random.randn(1000)
    >>> fig = plot_windowed_networks(x, window=50, step=10, method='hvg')
    >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive visualizations. "
            "Install with: pip install plotly"
        )
    
    from ts2net.multivariate.windows import ts_to_windows
    from ts2net.factory import create_graph_builder
    from ts2net.config import HVGConfig, NVGConfig, RecurrenceConfig, TransitionConfig
    
    # Extract windows
    windows = ts_to_windows(x, width=window, by=step)
    n_windows = windows.shape[0]
    
    # Create config factory
    def _create_hvg_config():
        return HVGConfig(
            enabled=True,
            output="edges",
            weighted=method_kwargs.get('weighted', False),
            weight_mode=method_kwargs.get('weight_mode'),
            limit=method_kwargs.get('limit'),
            directed=method_kwargs.get('directed', False)
        )
    
    def _create_nvg_config():
        return NVGConfig(
            enabled=True,
            output="edges",
            weighted=method_kwargs.get('weighted', False),
            weight_mode=method_kwargs.get('weight_mode'),
            limit=method_kwargs.get('limit'),
            max_edges=method_kwargs.get('max_edges'),
            max_edges_per_node=method_kwargs.get('max_edges_per_node'),
            max_memory_mb=method_kwargs.get('max_memory_mb')
        )
    
    def _create_recurrence_config():
        return RecurrenceConfig(
            enabled=True,
            output="edges",
            m=method_kwargs.get('m', 3),
            rule='knn',
            k=method_kwargs.get('k', 5),
            tau=method_kwargs.get('tau', 1),
            epsilon=method_kwargs.get('epsilon', 0.1),
            metric=method_kwargs.get('metric', 'euclidean')
        )
    
    def _create_transition_config():
        return TransitionConfig(
            enabled=True,
            output="edges",
            symbolizer=method_kwargs.get('symbolizer', 'ordinal'),
            order=method_kwargs.get('order', 3),
            n_states=method_kwargs.get('n_states')
        )
    
    config_map = {
        'hvg': _create_hvg_config,
        'nvg': _create_nvg_config,
        'recurrence': _create_recurrence_config,
        'transition': _create_transition_config,
    }
    
    config_factory = config_map.get(method.lower())
    if config_factory is None:
        raise ValueError(f"Unknown method: {method}. Must be one of {list(config_map.keys())}")
    
    # Build graphs for each window
    graphs = []
    timestamps = []
    
    for i, window_data in enumerate(windows):
        try:
            config = config_factory()
            builder = create_graph_builder(method, config, n_points=len(window_data))
            builder.build(window_data)
            G = builder.as_networkx(force=True)  # Force NetworkX conversion
            graphs.append(G)
            timestamps.append(f"Window {i+1} (t={i*step}:{i*step+window})")
        except Exception as e:
            # Skip failed windows
            import warnings
            warnings.warn(f"Failed to build graph for window {i}: {e}")
            continue
    
    if len(graphs) == 0:
        raise ValueError("No graphs were successfully built from windows")
    
    # Use first graph for position computation if not provided
    if pos is None:
        pos = nx.spring_layout(graphs[0], k=1, iterations=50)
    
    # Create visualization
    return plot_timeseries_network(
        graphs=graphs,
        timestamps=timestamps,
        pos=pos,
        title=f"Windowed {method.upper()} Networks (window={window}, step={step})",
    )


