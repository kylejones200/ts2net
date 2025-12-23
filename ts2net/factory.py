"""
Graph builder factory for creating network builders from configuration.

Uses dispatch dictionary pattern for clean, extensible graph creation.
"""

from __future__ import annotations

from typing import Dict, Any, Callable, Optional
import numpy as np
from numpy.typing import NDArray

from .api import HVG, NVG, RecurrenceNetwork, TransitionNetwork
from .config import HVGConfig, NVGConfig, RecurrenceConfig, TransitionConfig


def create_hvg_builder(config: HVGConfig) -> HVG:
    """Create HVG builder from configuration."""
    return HVG(
        weighted=config.weighted,
        limit=config.limit,
        output=config.output,
        directed=config.directed
    )


def create_nvg_builder(config: NVGConfig) -> NVG:
    """Create NVG builder from configuration."""
    return NVG(
        weighted=config.weighted,
        limit=config.limit,
        max_edges=config.max_edges,
        max_edges_per_node=config.max_edges_per_node,
        max_memory_mb=config.max_memory_mb,
        output=config.output
    )


def create_recurrence_builder(config: RecurrenceConfig, n_points: Optional[int] = None) -> RecurrenceNetwork:
    """Create RecurrenceNetwork builder from configuration."""
    # Safety check: refuse exact all-pairs for large n
    if n_points is not None and config.rule == 'epsilon' and n_points > 50_000:
        raise ValueError(
            f"Refusing exact all-pairs recurrence for n={n_points}. "
            f"Use rule='knn' with small k instead."
        )
    
    return RecurrenceNetwork(
        m=config.m,
        tau=config.tau,
        rule=config.rule,
        k=config.k,
        epsilon=config.epsilon,
        metric=config.metric,
        output=config.output
    )


def create_transition_builder(config: TransitionConfig) -> TransitionNetwork:
    """Create TransitionNetwork builder from configuration."""
    return TransitionNetwork(
        symbolizer=config.symbolizer,
        order=config.order,
        n_states=config.n_states,
        output=config.output
    )


# Dispatch dictionary for graph type -> builder factory
_BUILDER_FACTORIES: Dict[str, Callable] = {
    'hvg': create_hvg_builder,
    'nvg': create_nvg_builder,
    'recurrence': create_recurrence_builder,
    'transition': create_transition_builder,
}


def create_graph_builder(
    graph_type: str,
    config: HVGConfig | NVGConfig | RecurrenceConfig | TransitionConfig,
    n_points: Optional[int] = None
) -> HVG | NVG | RecurrenceNetwork | TransitionNetwork:
    """
    Create a graph builder from configuration using dispatch pattern.
    
    Parameters
    ----------
    graph_type : str
        Graph type: 'hvg', 'nvg', 'recurrence', or 'transition'
    config : GraphConfig
        Configuration object for the graph type
    n_points : int, optional
        Number of points in series (used for safety checks)
    
    Returns
    -------
    GraphBuilder
        Configured graph builder instance
    
    Raises
    ------
    ValueError
        If graph_type is unknown or configuration is invalid
    """
    factory = _BUILDER_FACTORIES.get(graph_type.lower())
    if factory is None:
        raise ValueError(f"Unknown graph type: {graph_type}. Must be one of {list(_BUILDER_FACTORIES.keys())}")
    
    # Recurrence builder needs n_points for safety check
    if graph_type.lower() == 'recurrence':
        return factory(config, n_points=n_points)
    else:
        return factory(config)


def build_graph_from_config(
    series: NDArray[np.float64],
    graph_type: str,
    config: HVGConfig | NVGConfig | RecurrenceConfig | TransitionConfig,
    include_triangles: bool = False
) -> Dict[str, Any]:
    """
    Build a graph from configuration and return statistics.
    
    Parameters
    ----------
    series : array
        Input time series
    graph_type : str
        Graph type: 'hvg', 'nvg', 'recurrence', or 'transition'
    config : GraphConfig
        Configuration object for the graph type
    include_triangles : bool
        Whether to include triangle counting in stats (computationally expensive)
    
    Returns
    -------
    dict
        Graph statistics dictionary
    """
    # Safety check: refuse dense adjacency unless explicitly forced
    force_dense = getattr(config, 'force_dense', False)
    if force_dense and len(series) > 50_000:
        raise ValueError(
            f"Refusing dense adjacency for n={len(series)}. "
            f"This would require ~{len(series)**2 * 8 / 1e9:.1f} GB. "
            f"Use sparse matrices or output='stats' instead."
        )
    
    builder = create_graph_builder(graph_type, config, n_points=len(series))
    graph = builder.build(series)
    
    # Get statistics
    stats = graph.stats(include_triangles=include_triangles)
    return stats


# Dispatch dictionary for aggregation functions
_AGGREGATE_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    'mean': lambda stats: stats.get('avg_degree', 0.0),
    'std': lambda stats: stats.get('std_degree', 0.0),
    'min': lambda stats: stats.get('min_degree', 0),
    'max': lambda stats: stats.get('max_degree', 0),
}


def aggregate_stats(stats: Dict[str, Any], aggregate: str) -> float:
    """
    Aggregate statistics using dispatch pattern.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary from graph builder
    aggregate : str
        Aggregation function: 'mean', 'std', 'min', 'max'
    
    Returns
    -------
    float
        Aggregated statistic value
    """
    func = _AGGREGATE_FUNCTIONS.get(aggregate.lower())
    if func is None:
        # Fallback: try to get directly from stats
        return stats.get(aggregate, 0.0)
    return func(stats)

