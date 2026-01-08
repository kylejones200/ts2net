"""
Built-in network metrics for time series networks.

Provides efficient computation of clustering, path lengths, and modularity
without requiring NetworkX conversion for basic metrics.
"""

from __future__ import annotations

from typing import Optional, Dict, Union
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

try:
    from networkx.algorithms import community
    HAS_COMMUNITY = True
except ImportError:
    HAS_COMMUNITY = False


def compute_clustering(
    G: Union[nx.Graph, nx.DiGraph],
    method: str = "average",
    sample_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute clustering coefficient metrics.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    method : str, default "average"
        Method: "average" (average clustering), "global" (transitivity),
        or "local" (returns per-node clustering)
    sample_size : int, optional
        For large graphs, sample nodes for local clustering computation
    
    Returns
    -------
    dict
        Dictionary with clustering metrics:
        - "avg_clustering": Average clustering coefficient
        - "transitivity": Global transitivity (triangles / triads)
        - "clustering_std": Standard deviation of local clustering (if method="local")
    """
    if G.number_of_nodes() == 0:
        return {
            "avg_clustering": np.nan,
            "transitivity": np.nan,
        }
    
    # Convert to undirected for clustering computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G
    
    results = {}
    
    # Average clustering
    if method in ("average", "local"):
        if sample_size and G_und.number_of_nodes() > sample_size:
            # Sample nodes for large graphs
            nodes = list(G_und.nodes())
            sampled = np.random.choice(nodes, size=sample_size, replace=False)
            local_clustering = nx.clustering(G_und, nodes=sampled)
            clustering_values = list(local_clustering.values())
            results["avg_clustering"] = float(np.mean(clustering_values))
            if method == "local":
                results["clustering_std"] = float(np.std(clustering_values))
        else:
            results["avg_clustering"] = float(nx.average_clustering(G_und))
            if method == "local":
                local_clustering = nx.clustering(G_und)
                clustering_values = list(local_clustering.values())
                results["clustering_std"] = float(np.std(clustering_values))
    else:
        results["avg_clustering"] = float(nx.average_clustering(G_und))
    
    # Global transitivity
    results["transitivity"] = float(nx.transitivity(G_und))
    
    return results


def compute_path_lengths(
    G: Union[nx.Graph, nx.DiGraph],
    method: str = "average",
    sample_size: Optional[int] = None,
    weight: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute shortest path length metrics.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    method : str, default "average"
        Method: "average" (average path length), "diameter" (longest shortest path),
        or "eccentricity" (returns per-node eccentricity)
    sample_size : int, optional
        For large graphs, sample node pairs for path computation
    weight : str, optional
        Edge attribute to use as weight (default: unweighted)
    
    Returns
    -------
    dict
        Dictionary with path length metrics:
        - "avg_path_length": Average shortest path length
        - "diameter": Graph diameter (longest shortest path)
        - "radius": Graph radius (minimum eccentricity)
        - "eccentricity_std": Standard deviation of eccentricity (if method="eccentricity")
    """
    if G.number_of_nodes() == 0:
        return {
            "avg_path_length": np.nan,
            "diameter": np.nan,
            "radius": np.nan,
        }
    
    # Convert to undirected for path computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G
    
    # Check connectivity
    if not nx.is_connected(G_und):
        # Use largest connected component
        largest_cc = max(nx.connected_components(G_und), key=len)
        G_und = G_und.subgraph(largest_cc).copy()
        if G_und.number_of_nodes() <= 1:
            return {
                "avg_path_length": np.nan,
                "diameter": np.nan,
                "radius": np.nan,
            }
    
    results = {}
    
    # Average shortest path length
    if sample_size and G_und.number_of_nodes() > sample_size:
        # Sample node pairs for large graphs
        nodes = list(G_und.nodes())
        sampled = np.random.choice(nodes, size=min(sample_size, len(nodes)), replace=False)
        path_lengths = []
        for i, u in enumerate(sampled):
            for v in sampled[i+1:]:
                try:
                    if weight:
                        length = nx.shortest_path_length(G_und, u, v, weight=weight)
                    else:
                        length = nx.shortest_path_length(G_und, u, v)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    continue
        results["avg_path_length"] = float(np.mean(path_lengths)) if path_lengths else np.nan
    else:
        try:
            if weight:
                results["avg_path_length"] = float(nx.average_shortest_path_length(G_und, weight=weight))
            else:
                results["avg_path_length"] = float(nx.average_shortest_path_length(G_und))
        except (nx.NetworkXError, nx.NetworkXNoPath):
            results["avg_path_length"] = np.nan
    
    # Diameter and radius
    try:
        if method in ("diameter", "eccentricity"):
            eccentricity = nx.eccentricity(G_und)
            ecc_values = list(eccentricity.values())
            results["diameter"] = float(max(ecc_values))
            results["radius"] = float(min(ecc_values))
            if method == "eccentricity":
                results["eccentricity_std"] = float(np.std(ecc_values))
        else:
            # Compute diameter only if needed
            results["diameter"] = float(nx.diameter(G_und))
            results["radius"] = float(nx.radius(G_und))
    except (nx.NetworkXError, nx.NetworkXNoPath):
        results["diameter"] = np.nan
        results["radius"] = np.nan
    
    return results


def compute_modularity(
    G: Union[nx.Graph, nx.DiGraph],
    method: str = "louvain",
    weight: Optional[str] = None,
    resolution: float = 1.0,
    seed: Optional[int] = None
) -> Dict[str, Union[float, dict]]:
    """
    Compute modularity and community structure.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    method : str, default "louvain"
        Community detection method: "louvain", "leiden", "greedy", or "label_propagation"
    weight : str, optional
        Edge attribute to use as weight (default: unweighted)
    resolution : float, default 1.0
        Resolution parameter for modularity (higher = more communities)
    seed : int, optional
        Random seed for community detection
    
    Returns
    -------
    dict
        Dictionary with modularity metrics:
        - "modularity": Modularity score
        - "n_communities": Number of communities detected
        - "communities": Dictionary mapping node to community ID (optional)
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {
            "modularity": np.nan,
            "n_communities": 0,
        }
    
    # Convert to undirected for modularity computation
    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G
    
    if not HAS_COMMUNITY:
        return {
            "modularity": np.nan,
            "n_communities": 0,
            "error": "networkx.algorithms.community not available"
        }
    
    results = {}
    
    # Detect communities
    if method == "louvain":
        communities_generator = community.louvain_communities(
            G_und, weight=weight, resolution=resolution, seed=seed
        )
        communities = list(communities_generator)
    elif method == "leiden":
        try:
            communities_generator = community.leiden_communities(
                G_und, weight=weight, resolution_parameter=resolution, seed=seed
            )
            communities = list(communities_generator)
        except AttributeError:
            # Fallback to louvain if leiden not available
            communities_generator = community.louvain_communities(
                G_und, weight=weight, resolution=resolution, seed=seed
            )
            communities = list(communities_generator)
    elif method == "greedy":
        # greedy_modularity_communities doesn't support seed or resolution parameters in all versions
        try:
            communities_generator = community.greedy_modularity_communities(
                G_und, weight=weight, resolution=resolution
            )
        except TypeError:
            # Fallback for older NetworkX versions
            communities_generator = community.greedy_modularity_communities(
                G_und, weight=weight
            )
        communities = list(communities_generator)
    elif method == "label_propagation":
        # label_propagation_communities doesn't support weight or seed parameters
        communities_dict = community.label_propagation_communities(G_und)
        # Convert to list of sets
        communities = [set(com) for com in communities_dict]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'louvain', 'leiden', 'greedy', or 'label_propagation'")
    
    # Compute modularity
    if weight:
        modularity = community.modularity(G_und, communities, weight=weight)
    else:
        modularity = community.modularity(G_und, communities)
    
    results["modularity"] = float(modularity)
    results["n_communities"] = len(communities)
    
    # Optionally include community assignments
    # (commented out by default to avoid large output)
    # community_dict = {}
    # for i, com in enumerate(communities):
    #     for node in com:
    #         community_dict[node] = i
    # results["communities"] = community_dict
    
    return results


def network_metrics(
    G: Union[nx.Graph, nx.DiGraph],
    include: Optional[list] = None,
    sample_size: Optional[int] = None,
    **kwargs
) -> Dict[str, Union[float, dict]]:
    """
    Compute comprehensive network metrics.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    include : list, optional
        Metrics to include: ["clustering", "path_lengths", "modularity"]
        If None, includes all metrics
    sample_size : int, optional
        For large graphs, sample nodes/pairs for expensive computations
    **kwargs
        Additional arguments passed to metric functions
    
    Returns
    -------
    dict
        Dictionary with all requested network metrics
    """
    if include is None:
        include = ["clustering", "path_lengths", "modularity"]
    
    results = {}
    
    if "clustering" in include:
        clustering_kwargs = {k: v for k, v in kwargs.items() if k in ["method", "sample_size"]}
        if sample_size:
            clustering_kwargs["sample_size"] = sample_size
        results.update(compute_clustering(G, **clustering_kwargs))
    
    if "path_lengths" in include:
        path_kwargs = {k: v for k, v in kwargs.items() if k in ["method", "sample_size", "weight"]}
        if sample_size:
            path_kwargs["sample_size"] = sample_size
        results.update(compute_path_lengths(G, **path_kwargs))
    
    if "modularity" in include:
        modularity_kwargs = {k: v for k, v in kwargs.items() if k in ["method", "weight", "resolution", "seed"]}
        results.update(compute_modularity(G, **modularity_kwargs))
    
    return results

