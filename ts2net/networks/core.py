"""
Core network construction algorithms.

This module provides functions to construct networks from time series data
using various algorithms, including k-nearest neighbors, ε-nearest neighbors,
and weighted networks.
"""

from typing import Optional, Sequence, Callable, Tuple
import numpy as np
import networkx as nx

# Optional dependencies
try:
    import pynndescent
except ImportError:
    pynndescent = None


def net_enn(
    D: np.ndarray, eps: float, names: Optional[Sequence[str]] = None
) -> nx.Graph:
    """
    Construct an ε-nearest neighbor (ε-NN) graph from a distance matrix.

    Args:
        D: Distance matrix of shape (n_samples, n_samples)
        eps: Distance threshold for connecting nodes
        names: Optional list of node names

    Returns:
        NetworkX graph representing the ε-NN network
    """
    n = D.shape[0]
    G = nx.Graph()

    # Add nodes
    if names is None:
        names = [f"n{i}" for i in range(n)]
    G.add_nodes_from(names)

    # Add edges
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                G.add_edge(names[i], names[j], weight=D[i, j])

    return G


def net_knn(
    D: np.ndarray,
    k: int,
    names: Optional[Sequence[str]] = None,
    symmetrize: bool = True,
) -> nx.Graph:
    """
    Construct a k-nearest neighbor (k-NN) graph from a distance matrix.

    Args:
        D: Distance matrix of shape (n_samples, n_samples)
        k: Number of nearest neighbors to connect
        names: Optional list of node names
        symmetrize: If True, ensure the graph is undirected

    Returns:
        NetworkX graph representing the k-NN network
    """
    n = D.shape[0]
    G = nx.Graph()

    # Add nodes
    if names is None:
        names = [f"n{i}" for i in range(n)]
    G.add_nodes_from(names)

    # For each node, find k nearest neighbors
    for i in range(n):
        # Get indices of k+1 smallest distances (including self)
        indices = np.argpartition(D[i], k + 1)[: k + 1]
        # Remove self from neighbors
        indices = [idx for idx in indices if idx != i][:k]

        # Add edges
        for j in indices:
            G.add_edge(names[i], names[j], weight=D[i, j])

    # Symmetrize the graph if needed
    if symmetrize and not G.is_directed():
        G = G.to_undirected()

    return G


def net_weighted(
    D: np.ndarray,
    fn: Optional[Callable[[float], float]] = None,
    names: Optional[Sequence[str]] = None,
) -> nx.Graph:
    """
    Construct a fully connected weighted graph from a distance matrix.

    Args:
        D: Distance matrix of shape (n_samples, n_samples)
        fn: Weight function (default: 1/(1+d))
        names: Optional list of node names

    Returns:
        Weighted NetworkX graph
    """
    if fn is None:
        fn = lambda d: 1.0 / (1.0 + d)

    n = D.shape[0]
    G = nx.Graph()

    # Add nodes
    if names is None:
        names = [f"n{i}" for i in range(n)]
    G.add_nodes_from(names)

    # Add weighted edges
    for i in range(n):
        for j in range(i + 1, n):
            weight = fn(D[i, j])
            G.add_edge(names[i], names[j], weight=weight)

    return G


def net_knn_approx(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    names: Optional[Sequence[str]] = None,
    symmetrize: bool = True,
) -> nx.Graph:
    """
    Construct an approximate k-NN graph using pynndescent for large datasets.

    Args:
        X: Input data array of shape (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric to use
        names: Optional list of node names
        symmetrize: If True, ensure the graph is undirected

    Returns:
        Approximate k-NN NetworkX graph
    """
    if pynndescent is None:
        raise ImportError("pynndescent is required for approximate nearest neighbors")

    n = X.shape[0]

    # Build the index and find nearest neighbors
    index = pynndescent.NNDescent(X, metric=metric)
    indices, distances = index.query(X, k=k + 1)  # k+1 because it includes self

    # Create graph
    G = nx.Graph()

    # Add nodes
    if names is None:
        names = [f"n{i}" for i in range(n)]
    G.add_nodes_from(names)

    # Add edges from nearest neighbors
    for i in range(n):
        for j, d in zip(indices[i][1:], distances[i][1:]):  # Skip self
            G.add_edge(names[i], names[j], weight=d)

    # Symmetrize the graph if needed
    if symmetrize and not G.is_directed():
        G = G.to_undirected()

    return G


def weight_inv1p(d: float) -> float:
    """Weight function: 1/(1+d)"""
    return 1.0 / (1.0 + float(d))


def net_enn_approx(
    X: np.ndarray,
    target_density: float = 0.05,
    metric: str = "euclidean",
    names: Optional[Sequence[str]] = None,
) -> Tuple[nx.Graph, float]:
    """
    Construct an approximate ε-NN graph with target density.

    Args:
        X: Input data array of shape (n_samples, n_features)
        target_density: Target edge density (0-1)
        metric: Distance metric to use
        names: Optional list of node names

    Returns:
        Tuple of (ε-NN graph, actual density)
    """
    n = X.shape[0]
    max_possible_edges = n * (n - 1) / 2
    target_edges = int(target_density * max_possible_edges)

    # Use binary search to find the right epsilon
    low = 0.0
    high = np.max(pdist(X, metric=metric))

    best_eps = high
    best_density = 1.0

    # Binary search for the right epsilon
    for _ in range(20):  # Limit iterations
        mid = (low + high) / 2

        # Count edges with current epsilon
        G = net_enn(squareform(pdist(X, metric=metric)), mid)
        density = G.number_of_edges() / max_possible_edges

        # Update search range
        if density < target_density:
            low = mid
        else:
            high = mid
            if density < best_density:
                best_density = density
                best_eps = mid

        # Early stopping if we're close enough
        if abs(density - target_density) < 0.01:
            break

    # Build final graph with best epsilon
    final_eps = best_eps
    G = net_enn(squareform(pdist(X, metric=metric)), final_eps, names=names)

    return G, G.number_of_edges() / max_possible_edges
