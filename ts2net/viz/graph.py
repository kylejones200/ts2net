"""
Unified graph visualization API for ts2net.

Provides TSGraph dataclass and builder functions for creating
visualization-ready graph objects with geometry and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform

from ts2net.core import embed as _embed


VisKind = Literal["hvg", "nvg", "bounded_nvg"]
WeightMode = Literal["none", "absdiff", "time_gap", "min_clearance", "slope"]
EpsMode = Literal["fraction_max", "percentile"]
Metric = Literal["euclidean", "sqeuclidean", "manhattan", "chebyshev"]
KnnMode = Literal["none", "mutual", "directed"]


@dataclass(frozen=True)
class TSGraph:
    """Container for a time-series-derived graph plus geometry and build metadata.

    Attributes:
        graph: NetworkX graph with node and edge attributes.
        pos: Optional 2D coordinates for nodes. Keys match graph nodes.
            Defaults to (t, x[t]) for visibility graphs.
        meta: Build metadata such as method, parameters, and data shape.
    """
    graph: Union[nx.Graph, nx.DiGraph]
    pos: Optional[Dict[int, np.ndarray]]
    meta: Dict[str, Any]


def build_visibility_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    kind: VisKind = "hvg",
    directed: bool = False,
    weighted: Union[bool, WeightMode] = False,
    weight_mode: Optional[WeightMode] = None,
    limit: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_edges_per_node: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    include_self_loops: bool = False,
    return_pos: bool = True,
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Construct HVG or NVG style graphs with optional direction and weights.

    Nodes map to time index i.
    Edge direction uses time forward orientation i -> j when i < j.
    Weights attach as edge attribute "weight".
    Distances and aux values attach as edge attributes when needed.

    Args:
        x: 1D series.
        kind: hvg, nvg, or bounded_nvg.
        directed: If True, emit a DiGraph and only time forward edges.
        weighted: False, True, or a string mode.
        weight_mode: Optional explicit mode. Overrides weighted when set.
        limit: Window limit for NVG variants.
        max_edges: Global cap for bounded_nvg.
        max_edges_per_node: Per node cap for bounded_nvg.
        max_memory_mb: Memory guard for bounded_nvg.
        include_self_loops: Rare. Default False.
        return_pos: If True, pos uses (t, x[t]) so plots match the series.
        dtype: Numeric dtype.

    Returns:
        TSGraph container with graph, pos, and meta.
    """
    x = np.asarray(x, dtype=dtype)
    n = int(x.shape[0])
    if n < 2:
        g = nx.DiGraph() if directed else nx.Graph()
        for i in range(n):
            g.add_node(i, t=i, x=float(x[i]))
        pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
        return TSGraph(graph=g, pos=pos, meta={"method": "visibility", "kind": kind})

    mode = _resolve_weight_mode(weighted=weighted, weight_mode=weight_mode)

    edges = _visibility_edges(
        x,
        kind=kind,
        limit=limit,
        max_edges=max_edges,
        max_edges_per_node=max_edges_per_node,
        max_memory_mb=max_memory_mb,
    )

    g: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if directed else nx.Graph()

    for i in range(n):
        g.add_node(i, t=i, x=float(x[i]))

    for i, j in edges:
        if not include_self_loops and i == j:
            continue
        u, v = (i, j) if not directed else ((i, j) if i < j else (j, i))
        if u == v and not include_self_loops:
            continue

        attrs: Dict[str, Any] = {}
        if mode != "none":
            attrs["weight"] = _vis_weight(x, i=u, j=v, mode=mode)
        g.add_edge(u, v, **attrs)

    pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
    meta = {
        "method": "visibility",
        "kind": kind,
        "directed": bool(directed),
        "weight_mode": mode,
        "limit": limit,
        "max_edges": max_edges,
        "max_edges_per_node": max_edges_per_node,
        "max_memory_mb": max_memory_mb,
        "n": n,
    }
    return TSGraph(graph=g, pos=pos, meta=meta)


def _resolve_weight_mode(*, weighted: Union[bool, WeightMode], weight_mode: Optional[WeightMode]) -> WeightMode:
    """Resolve weight mode from weighted flag and explicit weight_mode."""
    if weight_mode is not None:
        return weight_mode
    if weighted is True:
        return "absdiff"
    if weighted is False:
        return "none"
    return weighted


def _visibility_edges(
    x: np.ndarray,
    *,
    kind: VisKind,
    limit: Optional[int],
    max_edges: Optional[int],
    max_edges_per_node: Optional[int],
    max_memory_mb: Optional[int],
) -> np.ndarray:
    """Return array of undirected candidate edges as pairs (i, j)."""
    from ts2net.core.visibility.hvg import _hvg_edges_numba
    from ts2net.core.visibility.nvg import _nvg_edges_numba
    
    if kind == "hvg":
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=limit if limit else -1)
        edges = np.column_stack([rows, cols])
        # Normalize to (i, j) where i < j for consistency
        edges = np.sort(edges, axis=1)
        # Remove duplicates
        edges = np.unique(edges, axis=0)
        return edges
    
    if kind == "nvg":
        rows, cols, _, _ = _nvg_edges_numba(
            x, 
            weighted=False, 
            limit=limit if limit else -1,
            max_edges=max_edges if max_edges else -1,
            max_edges_per_node=max_edges_per_node if max_edges_per_node else -1
        )
        edges = np.column_stack([rows, cols])
        # Normalize to (i, j) where i < j for consistency
        edges = np.sort(edges, axis=1)
        # Remove duplicates
        edges = np.unique(edges, axis=0)
        return edges
    
    if kind == "bounded_nvg":
        # For now, use same as nvg with limits
        # TODO: Implement proper bounded_nvg with memory checks
        return _visibility_edges(x, kind="nvg", limit=limit, 
                                max_edges=max_edges, 
                                max_edges_per_node=max_edges_per_node,
                                max_memory_mb=max_memory_mb)
    
    raise ValueError(f"Unknown kind {kind}")


def _vis_weight(x: np.ndarray, *, i: int, j: int, mode: WeightMode) -> float:
    """Compute edge weight for visibility graph edge (i, j)."""
    if i == j:
        return 0.0
    if mode == "absdiff":
        return float(abs(x[j] - x[i]))
    if mode == "time_gap":
        return float(abs(j - i))
    if mode == "slope":
        return float((x[j] - x[i]) / (j - i))
    if mode == "min_clearance":
        return float(_min_clearance(x, i=i, j=j))
    if mode == "none":
        return 1.0
    raise ValueError(f"Unknown weight mode {mode}")


def _min_clearance(x: np.ndarray, *, i: int, j: int) -> float:
    """Compute minimum clearance between points i and j for visibility."""
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return float("inf")
    baseline = min(x[lo], x[hi])
    mid = x[lo + 1 : hi]
    if len(mid) == 0:
        return float("inf")
    return float(baseline - np.max(mid))


def build_recurrence_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    embed_dim: int = 3,
    delay: int = 1,
    eps: float = 0.2,
    eps_mode: EpsMode = "fraction_max",
    metric: Metric = "euclidean",
    exclude_diagonal: bool = True,
    theiler_window: int = 0,
    knn: int = 0,
    knn_mode: KnnMode = "none",
    weighted: bool = False,
    weight_mode: Literal["distance", "inverse_distance"] = "inverse_distance",
    return_pos: bool = True,
    node_id: Literal["time", "state"] = "time",
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Build an ε-recurrence network from a time series.

    You embed the series into state space, then connect nodes whose state vectors
    fall within an ε ball. This matches the style in recurrence-network figures
    where ε changes density.

    Args:
        x: 1D array-like of shape (n,) or array of shape (n, p) for multivariate.
        embed_dim: Embedding dimension m.
        delay: Delay τ in samples.
        eps: Threshold value. Interpreted by eps_mode.
        eps_mode: How to interpret eps.
            - "fraction_max": eps * max_pairwise_distance.
            - "percentile": eps is a percentile in [0, 100].
        metric: Distance metric in state space.
        exclude_diagonal: Remove self edges.
        theiler_window: Exclude edges for |i - j| <= theiler_window.
        knn: If > 0, also connect k nearest neighbors per node.
        knn_mode: How to apply knn edges.
            - "none": ignore knn parameter.
            - "mutual": keep only mutual kNN edges.
            - "directed": create directed kNN edges (returns DiGraph).
        weighted: Store weights on edges.
        weight_mode: Weight definition if weighted.
        return_pos: If True, return node positions as embedded vectors.
        node_id: Node labeling scheme.
            - "time": node id equals time index i.
            - "state": node id equals integer state index in embedding.
        dtype: Numeric dtype.

    Returns:
        TSGraph with:
            - graph nodes ordered by time index.
            - node attributes: "t" time index, "state" embedded vector.
            - edge attributes: "dist" and optionally "weight".
            - pos: embedded vectors (or None).
            - meta: method and parameters.
    """
    x = np.asarray(x, dtype=dtype)
    
    # Handle multivariate input (already embedded) vs 1D (needs embedding)
    if x.ndim == 1:
        # Embed 1D series
        embedded = _embed(x, m=embed_dim, tau=delay)
        n_original = len(x)
    else:
        # Already embedded or multivariate
        embedded = x
        n_original = len(x)
    
    n_embedded = len(embedded)
    
    # Compute pairwise distances
    if metric == "euclidean":
        dist_metric = "euclidean"
    elif metric == "sqeuclidean":
        dist_metric = "sqeuclidean"
    elif metric == "manhattan":
        dist_metric = "cityblock"
    elif metric == "chebyshev":
        dist_metric = "chebyshev"
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # For large n, use incremental edge building to avoid full distance matrix
    use_incremental = n_embedded > 5_000
    
    if use_incremental:
        # Build edges incrementally without storing full distance matrix
        D = None
        import warnings
        warnings.warn(
            f"Using incremental distance computation for n={n_embedded}. "
            f"This avoids storing the full distance matrix.",
            UserWarning
        )
    else:
        # For small n, compute full distance matrix (faster)
        D = cdist(embedded, embedded, metric=dist_metric)
    
    # Resolve epsilon threshold
    if use_incremental:
        # For incremental mode, we need to estimate max distance first
        # Sample a subset to estimate max distance
        sample_size = min(1000, n_embedded)
        sample_indices = np.random.choice(n_embedded, size=sample_size, replace=False)
        sample_embedded = embedded[sample_indices]
        sample_D = cdist(sample_embedded, sample_embedded, metric=dist_metric)
        sample_max_dist = np.max(sample_D[np.triu_indices_from(sample_D, k=1)])
        
        if eps_mode == "fraction_max":
            eps_threshold = eps * sample_max_dist
        elif eps_mode == "percentile":
            # Estimate percentile from sample
            sample_triu = sample_D[np.triu_indices_from(sample_D, k=1)]
            eps_threshold = np.percentile(sample_triu, eps)
        else:
            raise ValueError(f"Unknown eps_mode: {eps_mode}")
    else:
        # Use full distance matrix
        if eps_mode == "fraction_max":
            max_dist = np.max(D[np.triu_indices_from(D, k=1)])
            eps_threshold = eps * max_dist
        elif eps_mode == "percentile":
            # Get upper triangle (excluding diagonal)
            triu_dists = D[np.triu_indices_from(D, k=1)]
            eps_threshold = np.percentile(triu_dists, eps)
        else:
            raise ValueError(f"Unknown eps_mode: {eps_mode}")
    
    # Determine if we need a directed graph
    is_directed = (knn_mode == "directed")
    G: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if is_directed else nx.Graph()
    
    # Add nodes with attributes
    for i in range(n_embedded):
        node_attrs = {
            "t": i,  # Time index
            "state": embedded[i].copy(),  # Embedded state vector
        }
        G.add_node(i, **node_attrs)
    
    # Build epsilon-recurrence edges
    if use_incremental:
        # Incremental mode: compute distances on-the-fly
        for i in range(n_embedded):
            for j in range(i + 1, n_embedded):
                # Apply Theiler window
                if theiler_window > 0 and abs(i - j) <= theiler_window:
                    continue
                
                # Compute distance on-the-fly
                if metric == "euclidean":
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                elif metric == "sqeuclidean":
                    diff = embedded[i] - embedded[j]
                    dist = np.dot(diff, diff)
                elif metric == "manhattan":
                    dist = np.sum(np.abs(embedded[i] - embedded[j]))
                elif metric == "chebyshev":
                    dist = np.max(np.abs(embedded[i] - embedded[j]))
                else:
                    # Fallback to cdist for single pair
                    dist = cdist(embedded[i:i+1], embedded[j:j+1], metric=dist_metric)[0, 0]
                
                # Epsilon rule
                if dist <= eps_threshold:
                    edge_attrs: Dict[str, Any] = {"dist": float(dist)}
                    
                    if weighted:
                        if weight_mode == "distance":
                            edge_attrs["weight"] = float(dist)
                        elif weight_mode == "inverse_distance":
                            edge_attrs["weight"] = 1.0 / (1.0 + float(dist))
                        else:
                            raise ValueError(f"Unknown weight_mode: {weight_mode}")
                    
                    G.add_edge(i, j, **edge_attrs)
    else:
        # Full matrix mode: use precomputed distances
        for i in range(n_embedded):
            for j in range(i + 1, n_embedded):
                # Apply Theiler window
                if theiler_window > 0 and abs(i - j) <= theiler_window:
                    continue
                
                dist = D[i, j]
                
                # Epsilon rule
                if dist <= eps_threshold:
                    edge_attrs: Dict[str, Any] = {"dist": float(dist)}
                    
                    if weighted:
                        if weight_mode == "distance":
                            edge_attrs["weight"] = float(dist)
                        elif weight_mode == "inverse_distance":
                            edge_attrs["weight"] = 1.0 / (1.0 + float(dist))
                        else:
                            raise ValueError(f"Unknown weight_mode: {weight_mode}")
                    
                    G.add_edge(i, j, **edge_attrs)
    
    # Add kNN edges if requested
    if knn > 0 and knn_mode != "none":
        if use_incremental:
            # For large n, use approximate kNN if available
            try:
                from pynndescent import NNDescent
                # Build approximate kNN index
                index = NNDescent(embedded, metric=dist_metric, n_neighbors=knn+1)
                index.prepare()
            except ImportError:
                # Fall back to exact kNN (may be slow for very large n)
                index = None
        else:
            index = None
        
        for i in range(n_embedded):
            if index is not None:
                # Use approximate kNN
                neighbors, distances_knn = index.query(embedded[i:i+1], k=knn+1)
                # Remove self (first neighbor is usually self)
                neighbors = neighbors[0]
                distances_knn = distances_knn[0]
                k_nearest = [(int(neighbors[j]), float(distances_knn[j])) 
                            for j in range(len(neighbors)) 
                            if neighbors[j] != i][:knn]
            else:
                # Use exact kNN from distance matrix or compute on-the-fly
                if D is not None:
                    distances = [(j, D[i, j]) for j in range(n_embedded) if i != j]
                else:
                    # Compute distances on-the-fly
                    distances = []
                    for j in range(n_embedded):
                        if i != j:
                            if metric == "euclidean":
                                dist = np.linalg.norm(embedded[i] - embedded[j])
                            elif metric == "sqeuclidean":
                                diff = embedded[i] - embedded[j]
                                dist = np.dot(diff, diff)
                            elif metric == "manhattan":
                                dist = np.sum(np.abs(embedded[i] - embedded[j]))
                            elif metric == "chebyshev":
                                dist = np.max(np.abs(embedded[i] - embedded[j]))
                            else:
                                dist = cdist(embedded[i:i+1], embedded[j:j+1], metric=dist_metric)[0, 0]
                            distances.append((j, dist))
                distances.sort(key=lambda x: x[1])
                k_nearest = distances[:knn]
            
            if knn_mode == "mutual":
                # Only add if mutual
                for j, dist in k_nearest:
                    # Check if j also has i in its k nearest
                    if D is not None:
                        j_distances = [(k, D[j, k]) for k in range(n_embedded) if j != k]
                    else:
                        # Compute distances on-the-fly
                        j_distances = []
                        for k in range(n_embedded):
                            if j != k:
                                if metric == "euclidean":
                                    d = np.linalg.norm(embedded[j] - embedded[k])
                                elif metric == "sqeuclidean":
                                    diff = embedded[j] - embedded[k]
                                    d = np.dot(diff, diff)
                                elif metric == "manhattan":
                                    d = np.sum(np.abs(embedded[j] - embedded[k]))
                                elif metric == "chebyshev":
                                    d = np.max(np.abs(embedded[j] - embedded[k]))
                                else:
                                    d = cdist(embedded[j:j+1], embedded[k:k+1], metric=dist_metric)[0, 0]
                                j_distances.append((k, d))
                    j_distances.sort(key=lambda x: x[1])
                    j_k_nearest = [k for k, _ in j_distances[:knn]]
                    if i in j_k_nearest:
                        edge_attrs = {"dist": float(dist), "knn": True}
                        if weighted:
                            if weight_mode == "distance":
                                edge_attrs["weight"] = float(dist)
                            elif weight_mode == "inverse_distance":
                                edge_attrs["weight"] = 1.0 / (1.0 + float(dist))
                        G.add_edge(i, j, **edge_attrs)
            elif knn_mode == "directed":
                # Add directed edges
                for j, dist in k_nearest:
                    edge_attrs = {"dist": float(dist), "knn": True}
                    if weighted:
                        if weight_mode == "distance":
                            edge_attrs["weight"] = float(dist)
                        elif weight_mode == "inverse_distance":
                            edge_attrs["weight"] = 1.0 / (1.0 + float(dist))
                    G.add_edge(i, j, **edge_attrs)
    
    # Remove diagonal if requested
    if exclude_diagonal:
        G.remove_edges_from([(i, i) for i in range(n_embedded) if G.has_edge(i, i)])
    
    # Build position dictionary
    pos: Optional[Dict[int, np.ndarray]] = None
    if return_pos:
        if embed_dim == 2:
            # Use 2D embedding directly
            pos = {i: embedded[i] for i in range(n_embedded)}
        elif embed_dim > 2:
            # Project to 2D using PCA or first two dimensions
            # For simplicity, use first two dimensions
            pos = {i: embedded[i][:2] for i in range(n_embedded)}
        else:
            # 1D embedding - use (time, value) layout
            pos = {i: np.array([i, embedded[i][0]]) for i in range(n_embedded)}
    
    # Build metadata
    meta = {
        "method": "recurrence",
        "embed_dim": embed_dim,
        "delay": delay,
        "eps": eps,
        "eps_mode": eps_mode,
        "eps_threshold": float(eps_threshold),
        "metric": metric,
        "exclude_diagonal": exclude_diagonal,
        "theiler_window": theiler_window,
        "knn": knn,
        "knn_mode": knn_mode,
        "weighted": weighted,
        "weight_mode": weight_mode,
        "n_original": n_original,
        "n_embedded": n_embedded,
    }
    
    return TSGraph(graph=G, pos=pos, meta=meta)


def build_ordinal_partition_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    embed_dim: int = 4,
    delay: int = 1,
    directed: bool = True,
    weighted: bool = True,
    include_self_loops: bool = True,
    tie_break: Literal["stable", "jitter"] = "stable",
    return_pos: bool = False,
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Build an ordinal partition network.

    Nodes represent permutation patterns. Directed edges represent observed
    transitions between patterns. Edge weight equals count or probability.

    Args:
        x: 1D time series.
        embed_dim: Embedding dimension d (order of permutation).
        delay: Time delay τ.
        directed: If True, create directed graph (default True).
        weighted: If True, edge weights are transition counts.
        include_self_loops: If True, allow self-transitions.
        tie_break: How to handle ties in permutation patterns.
            - "stable": Use stable sort (preserves order of ties).
            - "jitter": Add small noise to break ties.
        return_pos: If True, compute 2D positions for nodes (default False).
        dtype: Numeric dtype.

    Returns:
        TSGraph with:
            - graph: DiGraph (or Graph if directed=False) with pattern nodes.
            - node attribute "pattern": tuple[int, ...] representing permutation.
            - node attribute "count": occurrence count of pattern.
            - edge attribute "weight": transition count (if weighted).
            - pos: Optional 2D positions for visualization.
            - meta: Build parameters and statistics.
    """
    x = np.asarray(x, dtype=dtype)
    n = len(x)
    
    if n < embed_dim:
        raise ValueError(f"Series too short: n={n} < embed_dim={embed_dim}")
    
    # Handle tie breaking
    if tie_break == "jitter":
        # Add tiny random noise to break ties
        noise = np.random.RandomState(42).uniform(-1e-10, 1e-10, size=n)
        x = x + noise
    
    # Embed time series
    embedded = _embed(x, m=embed_dim, tau=delay)
    n_windows = len(embedded)
    
    # Extract ordinal patterns
    patterns = []
    pattern_to_node = {}
    node_to_pattern = {}
    pattern_counts = {}
    
    for i, window in enumerate(embedded):
        # Get permutation pattern (argsort gives ranks)
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)
        
        # Track unique patterns
        if pattern not in pattern_to_node:
            node_id = len(pattern_to_node)
            pattern_to_node[pattern] = node_id
            node_to_pattern[node_id] = pattern
            pattern_counts[pattern] = 0
        
        pattern_counts[pattern] += 1
    
    # Build graph
    G: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if directed else nx.Graph()
    
    # Add nodes with attributes
    for pattern, node_id in pattern_to_node.items():
        G.add_node(
            node_id,
            pattern=pattern,
            count=pattern_counts[pattern],
        )
    
    # Build transition edges
    transition_counts: Dict[Tuple[int, int], int] = {}
    
    for i in range(n_windows - 1):
        source_pattern = patterns[i]
        target_pattern = patterns[i + 1]
        
        source_node = pattern_to_node[source_pattern]
        target_node = pattern_to_node[target_pattern]
        
        # Skip self-loops if not allowed
        if not include_self_loops and source_node == target_node:
            continue
        
        edge = (source_node, target_node)
        transition_counts[edge] = transition_counts.get(edge, 0) + 1
    
    # Add edges with weights
    for (source, target), count in transition_counts.items():
        edge_attrs: Dict[str, Any] = {}
        if weighted:
            edge_attrs["weight"] = count
        G.add_edge(source, target, **edge_attrs)
    
    # Compute positions if requested
    pos: Optional[Dict[int, np.ndarray]] = None
    if return_pos:
        # Use a simple layout: circular or spring
        if len(G.nodes()) <= 50:
            pos = nx.spring_layout(G, k=1.5, iterations=50)
        else:
            pos = nx.circular_layout(G)
    
    # Build metadata
    meta = {
        "method": "ordinal_partition",
        "embed_dim": embed_dim,
        "delay": delay,
        "directed": directed,
        "weighted": weighted,
        "include_self_loops": include_self_loops,
        "tie_break": tie_break,
        "n_original": n,
        "n_windows": n_windows,
        "n_patterns": len(pattern_to_node),
        "n_edges": G.number_of_edges(),
    }
    
    return TSGraph(graph=G, pos=pos, meta=meta)


def optimal_lag(x: np.ndarray, max_lag: int = 50) -> int:
    """Estimate optimal time delay τ using first zero of autocorrelation.
    
    This is a simple heuristic: find the first lag where autocorrelation
    crosses zero. If no zero crossing, return lag with minimum autocorrelation.
    
    Args:
        x: 1D time series.
        max_lag: Maximum lag to consider.
        
    Returns:
        Optimal delay τ (in samples).
    """
    x = np.asarray(x).flatten()
    n = len(x)
    max_lag = min(max_lag, n // 4)  # Don't exceed reasonable bounds
    
    # Compute autocorrelation
    x_centered = x - np.mean(x)
    autocorr = np.correlate(x_centered, x_centered, mode='full')
    autocorr = autocorr[n - 1:] / autocorr[n - 1]  # Normalize
    
    # Find first zero crossing
    for lag in range(1, min(max_lag, len(autocorr))):
        if autocorr[lag] <= 0:
            return lag
    
    # If no zero crossing, return lag with minimum autocorrelation
    min_idx = np.argmin(autocorr[1:max_lag]) + 1
    return min_idx


def optimal_dim(
    x: np.ndarray,
    delay: int = 1,
    dim_range: Tuple[int, int] = (2, 8),
) -> int:
    """Estimate optimal embedding dimension d by maximizing OPN degree variance.
    
    This heuristic builds ordinal partition networks for different dimensions
    and selects the dimension that maximizes variance in the degree distribution.
    Higher variance suggests richer structure.
    
    Args:
        x: 1D time series.
        delay: Time delay τ (use optimal_lag if unsure).
        dim_range: (min_dim, max_dim) to search.
        
    Returns:
        Optimal embedding dimension d.
    """
    x = np.asarray(x).flatten()
    min_dim, max_dim = dim_range
    
    if min_dim < 2:
        min_dim = 2
    if max_dim > 10:
        max_dim = 10  # Cap for computational reasons
    
    variances = []
    dims = []
    
    for d in range(min_dim, max_dim + 1):
        try:
            # Build OPN for this dimension
            tsgraph = build_ordinal_partition_graph(
                x,
                embed_dim=d,
                delay=delay,
                weighted=False,
                return_pos=False,
            )
            
            # Compute degree variance
            degrees = [tsgraph.graph.degree(n) for n in tsgraph.graph.nodes()]
            if len(degrees) > 1:
                var = float(np.var(degrees))
            else:
                var = 0.0
            
            variances.append(var)
            dims.append(d)
        except (ValueError, MemoryError):
            # Skip dimensions that fail
            continue
    
    if not variances:
        # Fallback to middle of range
        return (min_dim + max_dim) // 2
    
    # Return dimension with maximum variance
    best_idx = np.argmax(variances)
    return dims[best_idx]
