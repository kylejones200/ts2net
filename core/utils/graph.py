"""
Graph utility functions for network construction and analysis.

This module provides helper functions for working with graphs and adjacency matrices.
"""

from typing import Union, Optional

# Third-party imports
try:
    import networkx as nx
    from scipy import sparse
    import numpy as np
except ImportError:
    nx = None
    sparse = None
    np = None


def adj_to_graph(
    A: Union[np.ndarray, sparse.spmatrix], directed: bool = False
) -> Union[nx.Graph, nx.DiGraph]:
    """Convert an adjacency matrix to a networkx graph.

    Parameters:
    -----------
    A : array-like or sparse matrix, shape (n_nodes, n_nodes)
        Adjacency matrix of the graph.
    directed : bool, default=False
        If True, create a directed graph.

    Returns:
    --------
    G : networkx.Graph or networkx.DiGraph
        The constructed graph.
    """
    if nx is None:
        raise ImportError("networkx is required for graph conversion")

    if sparse is not None and sparse.issparse(A):
        # Convert sparse matrix to COO format for edge iteration
        A_coo = A.tocoo()

        # Create graph and add edges with weights
        G = nx.DiGraph() if directed else nx.Graph()

        # Add nodes first to ensure all nodes are included
        G.add_nodes_from(range(A.shape[0]))

        # Add edges with weights
        for i, j, w in zip(A_coo.row, A_coo.col, A_coo.data):
            if i != j or directed:  # Avoid duplicate edges in undirected graphs
                G.add_edge(i, j, weight=float(w))
    else:
        # Dense matrix
        G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)

    return G


def _giant_component(G: nx.Graph) -> nx.Graph:
    """Extract the giant component from a graph.

    Parameters:
    -----------
    G : networkx.Graph
        Input graph.

    Returns:
    --------
    H : networkx.Graph
        The largest connected component of G.
    """
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    # Get the largest component
    largest_component = max(components, key=len)

    # Create a subgraph with just the largest component
    H = G.subgraph(largest_component).copy()

    return H


def _sampled_combinations(nodes, r, max_samples=None, seed=3363):
    """Generate combinations of nodes, optionally with sampling.

    Parameters:
    -----------
    nodes : iterable
        The nodes to combine.
    r : int
        Size of combinations.
    max_samples : int, optional
        Maximum number of combinations to generate. If None, generate all.
    seed : int, default=3363
        Random seed for reproducibility.

    Returns:
    --------
    combinations : iterable
        Iterator over combinations of nodes.
    """
    import random
    from itertools import combinations, islice

    # If max_samples is None, return all combinations
    if max_samples is None:
        return combinations(nodes, r)

    # Otherwise, sample combinations
    n = len(nodes)
    total = math.comb(n, r)

    # If we want more samples than exist, just return all
    if max_samples >= total:
        return combinations(nodes, r)

    # Otherwise, sample without replacement
    random.seed(seed)
    indices = random.sample(range(total), max_samples)

    # Convert linear indices to combination indices
    # This is a simplified version that works for r=3
    # For general r, we'd need a more complex algorithm
    if r == 3:
        # For 3-node motifs, we can compute the indices directly
        # This is based on the combinatorial number system
        def index_to_comb(i):
            # Find the largest m such that C(m, 3) <= i
            m = 2
            while math.comb(m + 1, 3) <= i:
                m += 1
            i -= math.comb(m, 3)

            # Now find the largest n such that C(n, 2) <= i
            n = 1
            while math.comb(n + 1, 2) <= i:
                n += 1
            i -= math.comb(n, 2)

            return (i, n, m)

        # Get the sampled combinations
        sampled_indices = [index_to_comb(i) for i in indices]

        # Convert to node combinations
        node_list = list(nodes)
        return [
            (node_list[i], node_list[j], node_list[k]) for i, j, k in sampled_indices
        ]
    else:
        # For general r, we need to use a different approach
        # This is less efficient but works for any r
        all_combs = list(combinations(nodes, r))
        return [all_combs[i] for i in indices if i < len(all_combs)]


import math  # Moved here to be used in _sampled_combinations
