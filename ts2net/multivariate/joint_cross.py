"""
Joint and cross methods for multivariate time series network analysis.

This module provides methods for analyzing relationships between multiple
time series using network-based approaches:
- Joint recurrence networks: edges when multiple series are recurrent simultaneously
- Cross visibility graphs: visibility relationships between different series
- Coupling measures: synchronization and coupling strength metrics
- Multivariate network comparison: metrics for comparing multiple networks
"""

from __future__ import annotations

from typing import Tuple, Optional, List, Dict, Union
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix

from ..core.graph import Graph
from ..core.visibility.weights import compute_weight, WeightMode


def joint_recurrence_network(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    method: str = "epsilon",
    metric: str = "euclidean",
    weighted: bool = False,
    directed: bool = False
) -> Tuple[nx.Graph, NDArray]:
    """
    Construct a joint recurrence network from two time series.
    
    A joint recurrence occurs when both series are recurrent at the same time.
    An edge (i, j) exists if:
    - Series 1: points i and j are recurrent (within threshold or k-NN)
    - Series 2: points i and j are recurrent (within threshold or k-NN)
    
    Parameters
    ----------
    x1 : array (n,)
        First time series
    x2 : array (n,)
        Second time series (must have same length as x1)
    threshold : float, optional
        Distance threshold for epsilon recurrence (required if method="epsilon")
    k : int, optional
        Number of nearest neighbors for k-NN recurrence (required if method="knn")
    method : str, default "epsilon"
        Recurrence method: "epsilon" (threshold-based) or "knn" (k-nearest neighbors)
    metric : str, default "euclidean"
        Distance metric (only used for embedding if needed)
    weighted : bool, default False
        If True, weight edges by average distance
    directed : bool, default False
        If True, create directed graph
    
    Returns
    -------
    G : networkx.Graph or DiGraph
        Joint recurrence network
    A : array (n, n)
        Adjacency matrix
    
    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.random.randn(100)
    >>> x2 = np.random.randn(100)
    >>> G, A = joint_recurrence_network(x1, x2, threshold=0.5, method="epsilon")
    >>> print(f"Joint recurrence network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    """
    if len(x1) != len(x2):
        raise ValueError(f"Series must have same length: {len(x1)} != {len(x2)}")
    
    n = len(x1)
    
    # Build recurrence matrices for each series
    R1 = _build_recurrence_matrix(x1, threshold=threshold, k=k, method=method)
    R2 = _build_recurrence_matrix(x2, threshold=threshold, k=k, method=method)
    
    # Joint recurrence: both must be recurrent
    R_joint = R1.multiply(R2)  # Element-wise AND
    
    # Convert to networkx graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    G.add_nodes_from(range(n))
    
    # Add edges
    if weighted:
        # Use average distance as weight
        D1 = _pairwise_distances(x1)
        D2 = _pairwise_distances(x2)
        D_avg = (D1 + D2) / 2.0
        
        rows, cols = R_joint.nonzero()
        for i, j in zip(rows, cols):
            if not directed or i < j:
                weight = float(D_avg[i, j])
                G.add_edge(i, j, weight=weight)
    else:
        rows, cols = R_joint.nonzero()
        for i, j in zip(rows, cols):
            if not directed or i < j:
                G.add_edge(i, j)
    
    # Convert to dense adjacency matrix for return
    A = nx.adjacency_matrix(G, nodelist=range(n)).toarray()
    
    return G, A


def cross_visibility_graph(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    method: str = "hvg",
    weighted: Union[bool, str] = False,
    weight_mode: Optional[str] = None,
    limit: Optional[int] = None,
    directed: bool = False
) -> Tuple[nx.Graph, NDArray]:
    """
    Construct a cross visibility graph between two time series.
    
    A cross visibility graph connects points from different series if they
    are visible to each other. Visibility is determined by the visibility
    criterion applied across series boundaries.
    
    Parameters
    ----------
    x1 : array (n1,)
        First time series
    x2 : array (n2,)
        Second time series (can have different length)
    method : str, default "hvg"
        Visibility method: "hvg" (horizontal) or "nvg" (natural)
    weighted : bool or str, default False
        If True, use "absdiff" weight mode. If str, use that weight mode.
    weight_mode : str, optional
        Explicit weight mode (overrides weighted if provided)
    limit : int, optional
        Maximum temporal distance for visibility
    directed : bool, default False
        If True, create directed graph
    
    Returns
    -------
    G : networkx.Graph or DiGraph
        Cross visibility graph (bipartite: nodes 0..n1-1 from x1, n1..n1+n2-1 from x2)
    A : array (n1+n2, n1+n2)
        Adjacency matrix
    
    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.random.randn(50)
    >>> x2 = np.random.randn(50)
    >>> G, A = cross_visibility_graph(x1, x2, method="hvg")
    >>> print(f"Cross visibility: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    """
    n1, n2 = len(x1), len(x2)
    n_total = n1 + n2
    
    # Resolve weight mode
    if weight_mode is not None:
        w_mode = weight_mode
        is_weighted = True
    elif isinstance(weighted, str):
        w_mode = weighted
        is_weighted = True
    elif weighted is True:
        w_mode = "absdiff"
        is_weighted = True
    else:
        w_mode = None
        is_weighted = False
    
    # Create graph
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Add nodes: 0..n1-1 for x1, n1..n1+n2-1 for x2
    G.add_nodes_from(range(n_total))
    
    # Build cross visibility edges
    visibility_checkers = {
        "hvg": _cross_hvg_visible,
        "nvg": _cross_nvg_visible,
    }
    
    checker = visibility_checkers.get(method)
    if checker is None:
        raise ValueError(f"Unknown method: {method}. Use 'hvg' or 'nvg'")
    
    edges = []
    for i in range(n1):
        for j in range(n1, n_total):
            j_idx = j - n1
            
            if limit is not None and abs(i - j_idx) > limit:
                continue
            
            if checker(x1, x2, i, j_idx):
                if is_weighted:
                    weight = _compute_cross_weight(x1, x2, i, j_idx, w_mode)
                    edges.append((i, j, weight))
                else:
                    edges.append((i, j))
    
    # Add edges to graph
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    
    # Build adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=range(n_total)).toarray()
    
    return G, A


def _compute_cross_weight(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    i: int,
    j: int,
    w_mode: str
) -> float:
    """Compute weight for cross visibility edge."""
    weight_functions = {
        "absdiff": lambda: abs(x1[i] - x2[j]),
        "time_gap": lambda: abs(i - j),
        "slope": lambda: (x2[j] - x1[i]) / (j - i + 1),
    }
    func = weight_functions.get(w_mode, weight_functions["absdiff"])
    return float(func())


def _cross_hvg_visible(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    i: int,
    j: int
) -> bool:
    """Check if point i in x1 and point j in x2 are horizontally visible."""
    threshold = min(x1[i], x2[j])
    return (np.all(x1[i + 1:] < threshold) and np.all(x2[:j] < threshold))


def _cross_nvg_visible(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    i: int,
    j: int
) -> bool:
    """Check if point i in x1 and point j in x2 are naturally visible."""
    # For natural visibility, the line from (i, x1[i]) to (j, x2[j])
    # should not intersect intermediate points
    
    # Map to a common coordinate system
    # Use indices as x-coordinates, values as y-coordinates
    x_i, y_i = i, x1[i]
    x_j, y_j = len(x1) + j, x2[j]  # x2 starts after x1
    
    # Check intermediate points in x1 (after i)
    for k in range(i + 1, len(x1)):
        x_k, y_k = k, x1[k]
        line_y = y_i + (y_j - y_i) * (x_k - x_i) / (x_j - x_i)
        if y_k > line_y:
            return False
    
    # Check intermediate points in x2 (before j)
    for k in range(j):
        x_k, y_k = len(x1) + k, x2[k]
        line_y = y_i + (y_j - y_i) * (x_k - x_i) / (x_j - x_i)
        if y_k > line_y:
            return False
    
    return True


def coupling_strength(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    method: str = "joint_recurrence",
    threshold: Optional[float] = None,
    k: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute coupling strength between two time series.
    
    Parameters
    ----------
    x1 : array (n,)
        First time series
    x2 : array (n,)
        Second time series
    method : str, default "joint_recurrence"
        Coupling method: "joint_recurrence" or "cross_visibility"
    threshold : float, optional
        Threshold for recurrence (if method="joint_recurrence")
    k : int, optional
        k for k-NN recurrence (if method="joint_recurrence")
    
    Returns
    -------
    metrics : dict
        Dictionary with coupling metrics:
        - coupling_strength: Overall coupling strength (0-1)
        - joint_recurrence_rate: Fraction of joint recurrences
        - synchronization: Degree of synchronization
        - asymmetry: Asymmetry in coupling (0 = symmetric)
    
    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.random.randn(100)
    >>> x2 = x1 + 0.1 * np.random.randn(100)  # Coupled series
    >>> metrics = coupling_strength(x1, x2, method="joint_recurrence", threshold=0.5)
    >>> print(f"Coupling strength: {metrics['coupling_strength']:.3f}")
    """
    if len(x1) != len(x2):
        raise ValueError(f"Series must have same length: {len(x1)} != {len(x2)}")
    
    n = len(x1)
    
    if method == "joint_recurrence":
        # Build individual recurrence matrices
        R1 = _build_recurrence_matrix(x1, threshold=threshold, k=k, method="epsilon" if threshold else "knn")
        R2 = _build_recurrence_matrix(x2, threshold=threshold, k=k, method="epsilon" if threshold else "knn")
        
        # Joint recurrence
        R_joint = R1.multiply(R2)
        
        # Individual recurrence rates
        RR1 = R1.sum() / (n * n)
        RR2 = R2.sum() / (n * n)
        RR_joint = R_joint.sum() / (n * n)
        
        # Coupling strength: ratio of joint to expected (if independent)
        expected_joint = RR1 * RR2
        if expected_joint > 0:
            coupling_strength = RR_joint / expected_joint
        else:
            coupling_strength = 0.0
        
        # Synchronization: correlation of recurrence patterns
        R1_vec = R1.toarray().flatten()
        R2_vec = R2.toarray().flatten()
        synchronization = float(np.corrcoef(R1_vec, R2_vec)[0, 1])
        if np.isnan(synchronization):
            synchronization = 0.0
        
        # Asymmetry: difference in individual recurrence rates
        asymmetry = abs(RR1 - RR2)
        
        return {
            "coupling_strength": float(coupling_strength),
            "joint_recurrence_rate": float(RR_joint),
            "synchronization": float(synchronization),
            "asymmetry": float(asymmetry),
            "recurrence_rate_1": float(RR1),
            "recurrence_rate_2": float(RR2),
        }
    
    elif method == "cross_visibility":
        # Build cross visibility graph
        G, A = cross_visibility_graph(x1, x2, method="hvg")
        
        # Coupling strength: density of cross visibility
        n1, n2 = len(x1), len(x2)
        max_edges = n1 * n2
        actual_edges = G.number_of_edges()
        coupling_strength = actual_edges / max_edges if max_edges > 0 else 0.0
        
        # Additional metrics from cross visibility
        # Degree distribution analysis
        degrees_x1 = [G.degree(i) for i in range(n1)]
        degrees_x2 = [G.degree(i) for i in range(n1, n1 + n2)]
        
        return {
            "coupling_strength": float(coupling_strength),
            "cross_visibility_density": float(coupling_strength),
            "mean_degree_x1": float(np.mean(degrees_x1)),
            "mean_degree_x2": float(np.mean(degrees_x2)),
            "synchronization": float(np.corrcoef(degrees_x1, degrees_x2[:n1] if n1 == n2 else [0])[0, 1]) if n1 == n2 else 0.0,
            "asymmetry": float(abs(np.mean(degrees_x1) - np.mean(degrees_x2))),
        }
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'joint_recurrence' or 'cross_visibility'")


def network_comparison_metrics(
    networks: List[nx.Graph],
    names: Optional[List[str]] = None
) -> Dict[str, Union[float, NDArray]]:
    """
    Compute comparison metrics for multiple networks.
    
    Parameters
    ----------
    networks : list of networkx.Graph
        List of networks to compare
    names : list of str, optional
        Names for each network
    
    Returns
    -------
    metrics : dict
        Dictionary with comparison metrics:
        - density_similarity: Pairwise density correlations
        - degree_correlation: Pairwise degree sequence correlations
        - edge_overlap: Pairwise edge overlap (Jaccard similarity)
        - structural_similarity: Overall structural similarity matrix
    """
    n_networks = len(networks)
    
    if names is None:
        names = [f"network_{i}" for i in range(n_networks)]
    
    if len(names) != n_networks:
        raise ValueError(f"Number of names ({len(names)}) must match number of networks ({n_networks})")
    
    # Ensure all networks have same nodes
    all_nodes = set()
    for G in networks:
        all_nodes.update(G.nodes())
    all_nodes = sorted(all_nodes)
    
    # Compute metrics for each network
    densities = []
    degree_sequences = []
    edge_sets = []
    
    for G in networks:
        # Density
        n = G.number_of_nodes()
        m = G.number_of_edges()
        max_edges = n * (n - 1) / 2 if not G.is_directed() else n * (n - 1)
        density = m / max_edges if max_edges > 0 else 0.0
        densities.append(density)
        
        # Degree sequence (0 for nodes not in this network)
        degrees = np.array([G.degree(node) if node in G else 0 for node in all_nodes])
        degree_sequences.append(degrees)
        
        # Edge set (only edges that exist in this network)
        edge_sets.append(set(G.edges()))
    
    # Pairwise comparisons
    # Always create n_networks x n_networks matrix
    density_similarity = np.ones((n_networks, n_networks))
    
    if n_networks > 1:
        densities_array = np.array(densities)
        if np.std(densities_array) > 1e-10:  # Need variation to compute correlation
            try:
                corr_result = np.corrcoef(densities_array)
                # np.corrcoef always returns n x n for n inputs
                if corr_result.shape == (n_networks, n_networks):
                    density_similarity = corr_result
                # If shape is wrong (shouldn't happen), keep identity matrix
            except (ValueError, np.linalg.LinAlgError):
                # If correlation fails, use identity (all similar)
                pass
    
    degree_correlation = np.zeros((n_networks, n_networks))
    edge_overlap = np.zeros((n_networks, n_networks))
    
    for i in range(n_networks):
        for j in range(n_networks):
            # Degree correlation
            if np.std(degree_sequences[i]) > 0 and np.std(degree_sequences[j]) > 0:
                corr = np.corrcoef(degree_sequences[i], degree_sequences[j])[0, 1]
                degree_correlation[i, j] = corr if not np.isnan(corr) else 0.0
            else:
                degree_correlation[i, j] = 1.0 if np.allclose(degree_sequences[i], degree_sequences[j]) else 0.0
            
            # Edge overlap (Jaccard similarity)
            intersection = len(edge_sets[i] & edge_sets[j])
            union = len(edge_sets[i] | edge_sets[j])
            edge_overlap[i, j] = intersection / union if union > 0 else 0.0
    
    # Overall structural similarity (average of normalized metrics)
    structural_similarity = (
        (density_similarity + 1) / 2 +  # Normalize correlation to [0, 1]
        (degree_correlation + 1) / 2 +
        edge_overlap
    ) / 3.0
    
    return {
        "density_similarity": density_similarity,
        "degree_correlation": degree_correlation,
        "edge_overlap": edge_overlap,
        "structural_similarity": structural_similarity,
        "network_names": names,
    }


# Helper functions

def _build_recurrence_matrix(
    x: NDArray[np.float64],
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    method: str = "epsilon"
) -> csr_matrix:
    """Build recurrence matrix for a single series."""
    n = len(x)
    
    if method == "epsilon":
        if threshold is None:
            raise ValueError("threshold required for epsilon method")
        
        # Build sparse recurrence matrix
        rows, cols = [], []
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(x[i] - x[j])
                if dist <= threshold:
                    rows.append(i)
                    cols.append(j)
                    rows.append(j)
                    cols.append(i)
        
        data = np.ones(len(rows), dtype=float)
        R = csr_matrix((data, (rows, cols)), shape=(n, n))
        return R
    
    elif method == "knn":
        if k is None:
            k = 5  # Default k value
        
        # Build k-NN recurrence matrix
        rows, cols = [], []
        for i in range(n):
            # Compute distances
            distances = [(j, abs(x[i] - x[j])) for j in range(n) if i != j]
            distances.sort(key=lambda x: x[1])
            
            # Add k nearest neighbors
            for j, _ in distances[:k]:
                rows.append(i)
                cols.append(j)
        
        data = np.ones(len(rows), dtype=float)
        R = csr_matrix((data, (rows, cols)), shape=(n, n))
        return R
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _pairwise_distances(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute pairwise distances for a 1D series."""
    n = len(x)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = abs(x[i] - x[j])
    return D

