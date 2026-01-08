"""
Lightweight graph result object.

Inspired by ts2vg's clean API - keeps NetworkX optional, NumPy primary.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

# Import for type hints only
try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        import networkx as nx
except ImportError:
    pass


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
    _adjacency: Optional = None  # scipy sparse matrix, not dense
    _degrees: Optional[NDArray] = None
    _in_degrees: Optional[NDArray] = None
    _out_degrees: Optional[NDArray] = None
    
    @property
    def n_edges(self) -> int:
        """Number of edges"""
        # Support cached edge count for stats-only mode
        if hasattr(self, '_n_edges_cached'):
            return self._n_edges_cached
        return len(self.edges)
    
    def degree_sequence(self) -> NDArray[np.int64]:
        """
        Degree sequence (cached).
        
        For undirected graphs, returns total degree.
        For directed graphs, returns out-degree (for backward compatibility).
        
        Returns
        -------
        degrees : array (n_nodes,)
            Degree of each node (total for undirected, out-degree for directed)
        """
        if self.directed:
            return self.out_degree_sequence()
        else:
            if self._degrees is None:
                degrees = np.zeros(self.n_nodes, dtype=np.int64)
                for edge in self.edges:
                    i, j = edge[0], edge[1]
                    degrees[i] += 1
                    if i != j:
                        degrees[j] += 1
                self._degrees = degrees
            return self._degrees
    
    def in_degree_sequence(self) -> NDArray[np.int64]:
        """
        In-degree sequence for directed graphs (cached).
        
        Returns
        -------
        in_degrees : array (n_nodes,)
            In-degree of each node (number of incoming edges)
        """
        if not self.directed:
            raise ValueError("in_degree_sequence() only valid for directed graphs")
        if self._in_degrees is None:
            in_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                j = edge[1]  # Destination node
                in_degrees[j] += 1
            self._in_degrees = in_degrees
        return self._in_degrees
    
    def out_degree_sequence(self) -> NDArray[np.int64]:
        """
        Out-degree sequence for directed graphs (cached).
        
        Returns
        -------
        out_degrees : array (n_nodes,)
            Out-degree of each node (number of outgoing edges)
        """
        if not self.directed:
            raise ValueError("out_degree_sequence() only valid for directed graphs")
        if self._out_degrees is None:
            out_degrees = np.zeros(self.n_nodes, dtype=np.int64)
            for edge in self.edges:
                i = edge[0]  # Source node
                out_degrees[i] += 1
            self._out_degrees = out_degrees
        return self._out_degrees
    
    def adjacency_matrix(self, format: str = "sparse"):
        """
        Adjacency matrix (lazy, sparse by default).
        
        Parameters
        ----------
        format : str, default "sparse"
            Output format: "sparse" (CSR), "dense", or "coo"
        
        Returns
        -------
        A : scipy.sparse.csr_matrix, scipy.sparse.coo_matrix, or array
            Adjacency matrix. Sparse by default to avoid memory blowup.
        
        Raises
        ------
        ValueError
            If format="dense" and n_nodes > 50_000 (safety guardrail)
        """
        from scipy import sparse as sp
        
        # Safety guardrail: refuse dense for large graphs
        if format == "dense" and self.n_nodes > 50_000:
            raise ValueError(
                f"Refusing to build dense adjacency matrix for n={self.n_nodes} nodes. "
                f"This would require ~{self.n_nodes**2 * 8 / 1e9:.1f} GB of memory. "
                f"Use format='sparse' or format='coo' instead, or use only_degrees=True "
                f"to skip edge storage entirely."
            )
        
        # Build sparse COO from edges (memory efficient)
        if self._adjacency is None:
            if len(self.edges) == 0:
                # Empty graph
                self._adjacency = sp.coo_matrix((self.n_nodes, self.n_nodes))
            else:
                # Extract edge data
                if self.weighted:
                    rows = [e[0] for e in self.edges]
                    cols = [e[1] for e in self.edges]
                    data = [e[2] for e in self.edges]
                else:
                    rows = [e[0] for e in self.edges]
                    cols = [e[1] for e in self.edges]
                    data = [1.0] * len(self.edges)
                
                # Add reverse edges for undirected graphs
                if not self.directed:
                    # Create symmetric edges: add (j, i) for each (i, j)
                    reverse_rows = cols.copy()
                    reverse_cols = rows.copy()
                    rows = rows + reverse_rows
                    cols = cols + reverse_cols
                    data = data + data
                
                self._adjacency = sp.coo_matrix((data, (rows, cols)), 
                                                shape=(self.n_nodes, self.n_nodes))
        
        # Convert to requested format
        if format == "dense":
            return self._adjacency.toarray()
        elif format == "coo":
            return self._adjacency.tocoo()
        else:  # sparse (default)
            return self._adjacency.tocsr()
    
    def edges_coo(self) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """
        Return edges in COO format (coordinate arrays).
        
        Returns
        -------
        src : array (n_edges,)
            Source node indices
        dst : array (n_edges,)
            Destination node indices
        weight : array (n_edges,) or None
            Edge weights (if weighted), None otherwise
        """
        if len(self.edges) == 0:
            return (np.array([], dtype=np.int64), 
                    np.array([], dtype=np.int64), 
                    None if not self.weighted else np.array([]))
        
        # Handle different edge formats
        # Standard: (u, v) or (u, v, w)
        # Some networks might have nested tuples, handle gracefully
        try:
            if self.weighted:
                src = np.array([e[0] for e in self.edges], dtype=np.int64)
                dst = np.array([e[1] for e in self.edges], dtype=np.int64)
                weight = np.array([e[2] for e in self.edges], dtype=np.float64)
                return src, dst, weight
            else:
                src = np.array([e[0] for e in self.edges], dtype=np.int64)
                dst = np.array([e[1] for e in self.edges], dtype=np.int64)
                return src, dst, None
        except (IndexError, TypeError):
            # Handle edge format issues - try to extract from NetworkX edge format
            # This shouldn't happen with proper edge storage, but be defensive
            import warnings
            warnings.warn(
                f"Edge format issue in edges_coo(). Edges may be in unexpected format. "
                f"First edge: {self.edges[0] if self.edges else 'empty'}",
                UserWarning
            )
            # Fallback: return empty arrays
            return (np.array([], dtype=np.int64), 
                    np.array([], dtype=np.int64), 
                    None if not self.weighted else np.array([]))
    
    def as_networkx(self, force: bool = False):
        """
        Convert to NetworkX graph (optional dependency).
        
        Parameters
        ----------
        force : bool, default False
            If False, refuse conversion for n > 200_000 nodes (safety guardrail)
        
        Returns
        -------
        G : networkx.Graph or networkx.DiGraph
            NetworkX graph object
        
        Raises
        ------
        ImportError
            If NetworkX is not installed
        ValueError
            If n_nodes > 200_000 and force=False
        """
        # Safety guardrail for large graphs
        if not force and self.n_nodes > 200_000:
            raise ValueError(
                f"Refusing NetworkX conversion for n={self.n_nodes} nodes. "
                f"NetworkX is not designed for graphs this large. "
                f"Use force=True to override, or work with edges_coo() / "
                f"adjacency_matrix(format='sparse') instead."
            )
        
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
    
    def network_metrics(
        self,
        include: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Compute advanced network metrics (clustering, path lengths, modularity).
        
        Parameters
        ----------
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
            Dictionary with network metrics
        
        Examples
        --------
        >>> from ts2net import HVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> hvg = HVG()
        >>> hvg.build(x)
        >>> metrics = hvg._graph.network_metrics()
        >>> print(f"Clustering: {metrics['avg_clustering']:.3f}")
        >>> print(f"Avg path length: {metrics['avg_path_length']:.3f}")
        >>> print(f"Modularity: {metrics['modularity']:.3f}")
        """
        try:
            from ..networks.metrics import network_metrics
        except ImportError:
            raise ImportError(
                "Network metrics require networkx. Install with: pip install networkx"
            )
        
        G = self.as_networkx(force=(self.n_nodes <= 200_000))
        return network_metrics(G, include=include, sample_size=sample_size, **kwargs)
    
    def summary(self, include_triangles: bool = False) -> dict:
        """
        Graph summary statistics (computed from edges/degrees, no dense matrix).
        
        Parameters
        ----------
        include_triangles : bool, default False
            If True, compute triangle count (requires edge list, slower)
        
        Returns
        -------
        stats : dict
            Dictionary with n_nodes, n_edges, avg_degree, std_degree, density,
            and optionally triangles. For directed graphs, includes in/out degree
            statistics and irreversibility_score.
        """
        degrees = self.degree_sequence()
        max_edges = self.n_nodes * (self.n_nodes - 1)
        if not self.directed:
            max_edges //= 2
        
        stats = {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'avg_degree': float(np.mean(degrees)),
            'std_degree': float(np.std(degrees)) if len(degrees) > 1 else 0.0,
            'min_degree': int(np.min(degrees)) if len(degrees) > 0 else 0,
            'max_degree': int(np.max(degrees)) if len(degrees) > 0 else 0,
            'density': self.n_edges / max_edges if max_edges > 0 else 0.0,
        }
        
        # For directed graphs, add in/out degree statistics and irreversibility
        if self.directed:
            in_degrees = self.in_degree_sequence()
            out_degrees = self.out_degree_sequence()
            total_degrees = in_degrees + out_degrees
            
            stats['avg_in_degree'] = float(np.mean(in_degrees))
            stats['std_in_degree'] = float(np.std(in_degrees)) if len(in_degrees) > 1 else 0.0
            stats['avg_out_degree'] = float(np.mean(out_degrees))
            stats['std_out_degree'] = float(np.std(out_degrees)) if len(out_degrees) > 1 else 0.0
            stats['min_in_degree'] = int(np.min(in_degrees)) if len(in_degrees) > 0 else 0
            stats['max_in_degree'] = int(np.max(in_degrees)) if len(in_degrees) > 0 else 0
            stats['min_out_degree'] = int(np.min(out_degrees)) if len(out_degrees) > 0 else 0
            stats['max_out_degree'] = int(np.max(out_degrees)) if len(out_degrees) > 0 else 0
            
            # Irreversibility score: mean(abs(in_degree - out_degree) / total_degree)
            # For nodes with zero total degree, irreversibility is 0
            irreversibility = np.zeros(self.n_nodes, dtype=np.float64)
            mask = total_degrees > 0
            irreversibility[mask] = np.abs(in_degrees[mask] - out_degrees[mask]) / total_degrees[mask]
            stats['irreversibility_score'] = float(np.mean(irreversibility))
        
        if include_triangles and len(self.edges) > 0:
            # Count triangles from edge list (no dense matrix needed)
            triangles = self._count_triangles()
            stats['triangles'] = triangles
        
        # Add weight statistics if graph is weighted
        if self.weighted and len(self.edges) > 0:
            weights = []
            for edge in self.edges:
                if len(edge) == 3:
                    w = edge[2]
                    # Skip inf and nan values for statistics
                    if np.isfinite(w):
                        weights.append(w)
            
            if weights:
                weights_array = np.array(weights)
                stats['min_weight'] = float(np.min(weights_array))
                stats['max_weight'] = float(np.max(weights_array))
                stats['mean_weight'] = float(np.mean(weights_array))
                stats['std_weight'] = float(np.std(weights_array)) if len(weights_array) > 1 else 0.0
                # Count inf/nan weights separately
                inf_count = sum(1 for edge in self.edges 
                              if len(edge) == 3 and not np.isfinite(edge[2]))
                if inf_count > 0:
                    stats['inf_weight_count'] = inf_count
        
        return stats
    
    def _count_triangles(self) -> int:
        """Count triangles from edge list (memory efficient)."""
        if len(self.edges) == 0:
            return 0
        
        # Build neighbor sets (sparse representation)
        neighbors = {i: set() for i in range(self.n_nodes)}
        for edge in self.edges:
            i, j = edge[0], edge[1]
            neighbors[i].add(j)
            if not self.directed:
                neighbors[j].add(i)
        
        # Count triangles
        triangles = 0
        for i in range(self.n_nodes):
            for j in neighbors[i]:
                if j > i:  # Avoid double counting
                    # Count common neighbors
                    common = neighbors[i] & neighbors[j]
                    triangles += len(common)
        
        # Each triangle counted 3 times (once per edge)
        return triangles // 3 if not self.directed else triangles
    
    def __repr__(self) -> str:
        return f"Graph(n_nodes={self.n_nodes}, n_edges={self.n_edges}, directed={self.directed})"

