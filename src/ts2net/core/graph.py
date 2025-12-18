"""
Lightweight graph result object.

Inspired by ts2vg's clean API - keeps NetworkX optional, NumPy primary.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class Graph:
    """
    Lightweight graph representation.
    
    Primary storage is edges + optional adjacency matrix.
    NetworkX conversion is lazy and optional.
    
    Attributes
    ----------
    edges : list of (int, int) or (int, int, float)
        Edge list (unweighted or weighted)
    n_nodes : int
        Number of nodes
    directed : bool
        Whether graph is directed
    weighted : bool
        Whether edges have weights
    
    Examples
    --------
    >>> G = Graph(edges=[(0,1), (1,2)], n_nodes=3)
    >>> G.n_edges
    2
    >>> G.degree_sequence()
    array([1, 2, 1])
    >>> nx_graph = G.as_networkx()  # Optional conversion
    """
    
    edges: List[Tuple]
    n_nodes: int
    directed: bool = False
    weighted: bool = False
    _adjacency: Optional[NDArray] = None
    _degrees: Optional[NDArray] = None
    
    @property
    def n_edges(self) -> int:
        """Number of edges"""
        return len(self.edges)
    
    def degree_sequence(self) -> NDArray[np.int64]:
        """
        Degree sequence (cached).
        
        Returns
        -------
        degrees : array (n_nodes,)
            Degree of each node
        """
        if self._degrees is None:
            degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                i, j = edge[0], edge[1]
                degrees[i] += 1
                if not self.directed and i != j:
                    degrees[j] += 1
            self._degrees = degrees
        return self._degrees
    
    def adjacency_matrix(self, sparse: bool = False) -> NDArray:
        """
        Adjacency matrix (cached).
        
        Parameters
        ----------
        sparse : bool
            Return scipy sparse matrix (CSR format)
        
        Returns
        -------
        A : array (n_nodes, n_nodes) or scipy.sparse.csr_matrix
            Adjacency matrix
        """
        if self._adjacency is None:
            A = np.zeros((self.n_nodes, self.n_nodes))
            
            for edge in self.edges:
                if self.weighted:
                    i, j, w = edge
                else:
                    i, j = edge
                    w = 1.0
                
                A[i, j] = w
                if not self.directed:
                    A[j, i] = w
            
            self._adjacency = A
        
        if sparse:
            from scipy import sparse as sp
            return sp.csr_matrix(self._adjacency)
        
        return self._adjacency
    
    def as_networkx(self):
        """
        Convert to NetworkX graph (optional dependency).
        
        Returns
        -------
        G : networkx.Graph or networkx.DiGraph
            NetworkX graph object
        
        Raises
        ------
        ImportError
            If NetworkX is not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for as_networkx(). "
                "Install with: pip install networkx"
            )
        
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        G.add_nodes_from(range(self.n_nodes))
        
        if self.weighted:
            G.add_weighted_edges_from(self.edges)
        else:
            G.add_edges_from(self.edges)
        
        return G
    
    def summary(self) -> dict:
        """
        Graph summary statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with n_nodes, n_edges, avg_degree, density
        """
        degrees = self.degree_sequence()
        max_edges = self.n_nodes * (self.n_nodes - 1)
        if not self.directed:
            max_edges //= 2
        
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'avg_degree': float(np.mean(degrees)),
            'density': self.n_edges / max_edges if max_edges > 0 else 0.0,
        }
    
    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, directed={self.directed})"

