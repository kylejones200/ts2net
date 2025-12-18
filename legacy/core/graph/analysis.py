"""
Graph analysis functions for network metrics and properties.

This module provides functions for analyzing graph properties, including
small-world properties, motif analysis, and graph summaries.
"""

from typing import Union, Optional, Dict, Any, Tuple
import math
import random
import numpy as np

# Third-party imports
try:
    import networkx as nx
except ImportError:
    nx = None

# Local imports
from ..utils.graph import _sampled_combinations, _giant_component


def triangle_count(G: nx.Graph) -> int:
    """Count the number of triangles in a graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.

    Returns:
    --------
    int
        Number of triangles in the graph.
    """
    if nx is None:
        raise ImportError("networkx is required for triangle_count")

    # For small graphs, use networkx's built-in function
    if len(G) < 1000:
        return sum(nx.triangles(G).values()) // 3

    # For large graphs, use a more memory-efficient approach
    triangles = 0
    for v in G:
        neighbors = set(G[v])
        for u in neighbors:
            if u > v:  # Avoid double counting
                triangles += len(neighbors.intersection(G[u]))
    return triangles // 3


def wedge_count(G: nx.Graph) -> int:
    """Count the number of wedges (2-paths) in a graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.

    Returns:
    --------
    int
        Number of wedges in the graph.
    """
    if nx is None:
        raise ImportError("networkx is required for wedge_count")

    return sum(d * (d - 1) // 2 for _, d in G.degree())


def _deg_seq(sub: nx.Graph) -> Tuple[int, ...]:
    """Get the degree sequence of a graph as a sorted tuple."""
    return tuple(sorted((d for _, d in sub.degree()), reverse=True))


def directed_3node_motifs(
    G: nx.DiGraph, max_samples: Optional[int] = None, seed: int = 3363
) -> Dict[Tuple[int, int, int], int]:
    """Count directed 3-node motifs in a directed graph.

    Parameters:
    -----------
    G : networkx.DiGraph
        Input directed graph.
    max_samples : int, optional
        Maximum number of node triplets to sample. If None, use all triplets.
    seed : int, default=3363
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary mapping motif types to their counts.
    """
    if nx is None:
        raise ImportError("networkx is required for directed_3node_motifs")

    # All possible 3-node directed graphlets (unlabeled)
    # Each is represented by a tuple of edge directions (0=no edge, 1=forward, -1=backward)
    motifs = {
        (0, 0, 0, 0, 0, 0): 0,  # No edges
        (1, 0, 0, 0, 0, 0): 0,  # One edge
        (1, 0, 0, 1, 0, 0): 0,  # Two edges, no triangle
        (1, 1, 0, 0, 0, 0): 0,  # Two edges, no triangle, different direction
        (1, 1, 1, 0, 0, 0): 0,  # Three edges, triangle
        (1, 1, 1, 1, 0, 0): 0,  # Three edges, triangle with reciprocal
        (1, 1, 1, 1, 1, 1): 0,  # Complete graph
    }

    nodes = list(G.nodes())
    n = len(nodes)

    # Sample node triplets
    triplets = _sampled_combinations(nodes, 3, max_samples, seed)

    for u, v, w in triplets:
        # Get all possible directed edges
        edges = [
            (u, v),
            (v, u),  # u-v
            (v, w),
            (w, v),  # v-w
            (u, w),
            (w, u),  # u-w
        ]

        # Check which edges exist
        edge_flags = [1 if G.has_edge(*e) else 0 for e in edges]

        # Find matching motif
        for motif in motifs:
            if all(a == b for a, b in zip(edge_flags, motif[:6])):
                motifs[motif] += 1
                break

    return motifs


def undirected_4node_motifs(
    G: nx.Graph, max_samples: Optional[int] = None, seed: int = 3363
) -> Dict[Tuple[int, ...], int]:
    """Count undirected 4-node motifs in a graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input undirected graph.
    max_samples : int, optional
        Maximum number of node quadruplets to sample. If None, use all.
    seed : int, default=3363
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary mapping degree sequences to motif counts.
    """
    if nx is None:
        raise ImportError("networkx is required for undirected_4node_motifs")

    motifs = {}
    nodes = list(G.nodes())
    n = len(nodes)

    # Sample node quadruplets
    quads = _sampled_combinations(nodes, 4, max_samples, seed)

    for quad in quads:
        # Create subgraph
        sub = G.subgraph(quad).copy()

        # Get degree sequence as a tuple
        deg_seq = _deg_seq(sub)

        # Update count
        if deg_seq in motifs:
            motifs[deg_seq] += 1
        else:
            motifs[deg_seq] = 1

    return motifs


def small_world_summary(G: Union[nx.Graph, nx.DiGraph]) -> Dict[str, float]:
    """Calculate small-world properties of a graph.

    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'avg_shortest_path': Average shortest path length
        - 'clustering_coefficient': Average clustering coefficient
        - 'small_world_omega': Small-world coefficient omega
        - 'small_world_sigma': Small-world coefficient sigma
    """
    if nx is None:
        raise ImportError("networkx is required for small_world_summary")

    # Get the largest connected component
    G_giant = _giant_component(G)

    # Calculate average shortest path length
    try:
        avg_path = nx.average_shortest_path_length(G_giant)
    except nx.NetworkXError:
        # Graph is not connected
        avg_path = float("nan")

    # Calculate average clustering coefficient
    clustering = nx.average_clustering(G)

    # Calculate small-world coefficients
    n = len(G)
    k = sum(d for _, d in G.degree()) / n  # Average degree

    # Expected clustering for random graph
    C_rand = k / n if n > 1 else 0

    # Expected path length for random graph
    if k > 1 and n > 1:
        L_rand = math.log(n) / math.log(k)
    else:
        L_rand = float("nan")

    # Small-world coefficients
    omega = (L_rand / avg_path) - (clustering / C_rand) if C_rand > 0 else float("nan")
    sigma = (
        (clustering / C_rand) / (avg_path / L_rand)
        if C_rand > 0 and L_rand > 0
        else float("nan")
    )

    return {
        "avg_shortest_path": avg_path,
        "clustering_coefficient": clustering,
        "small_world_omega": omega,
        "small_world_sigma": sigma,
    }


def graph_summary(
    G: Union[nx.Graph, nx.DiGraph],
    motifs: Optional[str] = None,
    motif_samples: Optional[int] = None,
    seed: int = 3363,
) -> Dict[str, Any]:
    """Generate a summary of graph properties.

    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    motifs : str, optional
        Type of motifs to count ('3node' or '4node').
    motif_samples : int, optional
        Maximum number of samples to use for motif counting.
    seed : int, default=3363
        Random seed for reproducibility.

    Returns:
    --------
    dict
        Dictionary containing graph properties.
    """
    if nx is None:
        raise ImportError("networkx is required for graph_summary")

    # Basic graph properties
    summary = {
        "n_nodes": len(G),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G) if len(G) > 1 else 0,
        "is_directed": G.is_directed(),
    }

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    if degrees:
        summary.update(
            {
                "avg_degree": sum(degrees) / len(degrees),
                "min_degree": min(degrees),
                "max_degree": max(degrees),
            }
        )

    # Small-world properties
    small_world = small_world_summary(G)
    summary.update(small_world)

    # Motif analysis if requested
    if motifs == "3node" and G.is_directed():
        summary["motifs_3node"] = directed_3node_motifs(G, motif_samples, seed)
    elif motifs == "4node" and not G.is_directed():
        summary["motifs_4node"] = undirected_4node_motifs(G, motif_samples, seed)

    return summary
