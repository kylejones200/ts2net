"""
Wraps existing implementations with new lightweight Graph interface.
All old functionality preserved via .fit_transform().
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from .core.graph import Graph

# Import existing implementations from ts2net.core
from .core.visibility import HVG as _HVG_Old
from .core.visibility import NVG as _NVG_Old
from .core.recurrence import RecurrenceNetwork as _RN_Old
from .core.transition import TransitionNetwork as _TN_Old


class HVG:
    """
    Horizontal Visibility Graph.
    
    
    
    Parameters
    ----------
    weighted : bool
        Edge weights = abs(y_i - y_j)
    limit : int, optional
        Maximum temporal distance
    only_degrees : bool
        Performance mode - skip edge storage
    
    Examples
    --------
    >>> import numpy as np
    >>> from ts2net import HVG
    >>> x = np.random.randn(1000)
    >>> hvg = HVG()
    >>> hvg.build(x)
    >>> print(hvg.n_nodes, hvg.n_edges)
    >>> degrees = hvg.degree_sequence()
    >>> A = hvg.adjacency_matrix()
    >>> G_nx = hvg.as_networkx()  # Optional
    """
    
    def __init__(self, weighted: bool = False, limit: Optional[int] = None, 
                 only_degrees: bool = False, output: str = "edges"):
        """
        Parameters
        ----------
        weighted : bool
            Edge weights = abs(y_i - y_j)
        limit : int, optional
            Maximum temporal distance
        only_degrees : bool
            DEPRECATED: Use output="degrees" instead. Performance mode - skip edge storage
        output : str, default "edges"
            Output mode: "edges" (full edge list), "degrees" (degree sequence only),
            or "stats" (summary statistics only, most memory efficient)
        """
        self.weighted = weighted
        self.limit = limit
        # Backward compatibility
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        self._impl = _HVG_Old(weighted=weighted, limit=limit)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build HVG from time series."""
        # Use old implementation
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert to new Graph object based on output mode
        if self.output == "degrees":
            # Degrees-only mode: compute degrees, skip edges
            degrees = np.array([d for _, d in G_nx.degree()])
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None,  # Don't store sparse matrix either
                _degrees=degrees
            )
        elif self.output == "stats":
            # Stats-only mode: compute degrees and basic stats, skip edges
            degrees = np.array([d for _, d in G_nx.degree()])
            n_edges = G_nx.number_of_edges()
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None,
                _degrees=degrees
            )
            # Store edge count separately since we don't have edges
            self._graph._n_edges_cached = n_edges
        else:  # output == "edges"
            # Full edge list mode
            edges = list(G_nx.edges(data='weight' if self.weighted else False))
            if self.weighted:
                edges = [(u, v, w) for u, v, w in edges]
            
            self._graph = Graph(
                edges=edges,
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None  # Build sparse lazily if needed
            )
        
        return self
    
    @property
    def edges(self):
        """Edge list (None if output='degrees' or 'stats')"""
        if self._graph is None:
            raise ValueError("Call build() first")
        if self.output in ("degrees", "stats"):
            return None
        return self._graph.edges
    
    @property
    def n_nodes(self):
        """Number of nodes"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_nodes
    
    @property
    def n_edges(self):
        """Number of edges"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_edges
    
    def degree_sequence(self):
        """Degree sequence"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def stats(self, include_triangles: bool = False) -> dict:
        """Summary statistics (memory efficient, no dense matrix)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.summary(include_triangles=include_triangles)
    
    def adjacency_matrix(self, format: str = "sparse"):
        """Adjacency matrix (sparse by default)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(format=format)
    
    def edges_coo(self):
        """Return edges in COO format (src, dst, weight arrays)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.edges_coo()
    
    def as_networkx(self, force: bool = False):
        """Convert to NetworkX (refuses for n > 200k unless force=True)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx(force=force)


class NVG:
    """Natural Visibility Graph"""
    
    def __init__(self, weighted: bool = False, limit: Optional[int] = None,
                 only_degrees: bool = False, output: str = "edges",
                 max_edges: Optional[int] = None, max_edges_per_node: Optional[int] = None,
                 max_memory_mb: Optional[float] = None):
        """
        Parameters
        ----------
        weighted : bool
            Edge weights = abs(y_i - y_j)
        limit : int, optional
            Maximum temporal distance (horizon limit, critical scale control)
        only_degrees : bool
            DEPRECATED: Use output="degrees" instead
        output : str, default "edges"
            Output mode: "edges", "degrees", or "stats"
        max_edges : int, optional
            Maximum total edges (safety cap)
        max_edges_per_node : int, optional
            Maximum edges per node (additional scale control)
        max_memory_mb : float, optional
            Maximum memory in MB (converted to max_edges estimate)
        """
        self.weighted = weighted
        self.limit = limit
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        self._impl = _NVG_Old(
            weighted=weighted, limit=limit,
            max_edges=max_edges, max_edges_per_node=max_edges_per_node,
            max_memory_mb=max_memory_mb
        )
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build NVG from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert based on output mode (same pattern as HVG)
        if self.output == "degrees":
            degrees = np.array([d for _, d in G_nx.degree()])
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None,
                _degrees=degrees
            )
        elif self.output == "stats":
            degrees = np.array([d for _, d in G_nx.degree()])
            n_edges = G_nx.number_of_edges()
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None,
                _degrees=degrees
            )
            self._graph._n_edges_cached = n_edges
        else:  # output == "edges"
            edges = list(G_nx.edges(data='weight' if self.weighted else False))
            if self.weighted:
                edges = [(u, v, w) for u, v, w in edges]
            
            self._graph = Graph(
                edges=edges,
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=None
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        if self.output in ("degrees", "stats"):
            return None
        return self._graph.edges
    
    @property
    def n_nodes(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_nodes
    
    @property
    def n_edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_edges
    
    def degree_sequence(self):
        """Degree sequence"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def stats(self, include_triangles: bool = False) -> dict:
        """Summary statistics (memory efficient, no dense matrix)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.summary(include_triangles=include_triangles)
    
    def adjacency_matrix(self, format: str = "sparse"):
        """Adjacency matrix (sparse by default)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(format=format)
    
    def edges_coo(self):
        """Return edges in COO format (src, dst, weight arrays)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.edges_coo()
    
    def as_networkx(self, force: bool = False):
        """Convert to NetworkX (refuses for n > 200k unless force=True)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx(force=force)


class RecurrenceNetwork:
    """Recurrence Network"""
    
    def __init__(self, m: Optional[int] = None, tau: int = 1, rule: str = 'knn',
                 k: int = 5, epsilon: float = 0.1, metric: str = 'euclidean',
                 only_degrees: bool = False, output: str = "edges"):
        """
        Parameters
        ----------
        m : int, optional
            Embedding dimension
        tau : int
            Time delay
        rule : str
            'knn', 'epsilon', 'radius'
        k : int
            Neighbors for k-NN
        epsilon : float
            Threshold for epsilon-recurrence
        metric : str
            Distance metric
        only_degrees : bool
            DEPRECATED: Use output="degrees" instead
        output : str, default "edges"
            Output mode: "edges", "degrees", or "stats"
        """
        self.m = m
        self.tau = tau
        self.rule = rule
        self.k = k
        self.epsilon = epsilon
        self.metric = metric
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        # Map parameters to old implementation (uses threshold instead of epsilon)
        # For k-NN rule, use k parameter; for epsilon rule, use threshold
        threshold = epsilon if rule == 'epsilon' else None
        self._impl = _RN_Old(m=m, tau=tau, rule=rule, k=k, metric=metric, threshold=threshold)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build recurrence network from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert based on output mode
        if self.output == "degrees":
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=False,
                weighted=False,
                _adjacency=None,
                _degrees=degrees
            )
        elif self.output == "stats":
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            n_edges = G_nx.number_of_edges()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=False,
                weighted=False,
                _adjacency=None,
                _degrees=degrees
            )
            self._graph._n_edges_cached = n_edges
        else:  # output == "edges"
            edges = list(G_nx.edges())
            self._graph = Graph(
                edges=edges,
                n_nodes=G_nx.number_of_nodes(),
                directed=False,
                weighted=False,
                _adjacency=None
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        if self.output in ("degrees", "stats"):
            return None
        return self._graph.edges
    
    @property
    def n_nodes(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_nodes
    
    @property
    def n_edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_edges
    
    def degree_sequence(self):
        """Degree sequence"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def stats(self, include_triangles: bool = False) -> dict:
        """Summary statistics (memory efficient, no dense matrix)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.summary(include_triangles=include_triangles)
    
    def adjacency_matrix(self, format: str = "sparse"):
        """Adjacency matrix (sparse by default)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(format=format)
    
    def edges_coo(self):
        """Return edges in COO format (src, dst, weight arrays)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.edges_coo()
    
    def as_networkx(self, force: bool = False):
        """Convert to NetworkX (refuses for n > 200k unless force=True)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx(force=force)


class TransitionNetwork:
    """Transition Network"""
    
    def __init__(self, symbolizer: str = 'ordinal', order: int = 3, delay: int = 1,
                 tie_rule: str = 'stable', bins: int = 5, normalize: bool = True,
                 sparse: bool = False, only_degrees: bool = False, output: str = "edges"):
        """
        Parameters
        ----------
        symbolizer : str
            Symbolization method
        order : int
            Order of patterns
        delay : int
            Time delay
        tie_rule : str
            How to handle ties
        bins : int
            Number of bins
        normalize : bool
            Normalize input
        sparse : bool
            DEPRECATED: Adjacency matrices are always sparse now
        only_degrees : bool
            DEPRECATED: Use output="degrees" instead
        output : str, default "edges"
            Output mode: "edges", "degrees", or "stats"
        """
        self.symbolizer = symbolizer
        self.order = order
        self.delay = delay
        self.tie_rule = tie_rule
        self.bins = bins
        self.normalize = normalize
        self.sparse = sparse
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        # Map parameters to old implementation (doesn't accept normalize or sparse)
        self._impl = _TN_Old(
            symbolizer=symbolizer, order=order, delay=delay,
            tie_rule=tie_rule, bins=bins
        )
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build transition network from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert based on output mode
        if self.output == "degrees":
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=True,  # Transition networks are directed
                weighted=False,
                _adjacency=None,
                _degrees=degrees
            )
        elif self.output == "stats":
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            n_edges = G_nx.number_of_edges()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=True,
                weighted=False,
                _adjacency=None,
                _degrees=degrees
            )
            self._graph._n_edges_cached = n_edges
        else:  # output == "edges"
            edges = list(G_nx.edges())
            self._graph = Graph(
                edges=edges,
                n_nodes=G_nx.number_of_nodes(),
                directed=True,
                weighted=False,
                _adjacency=None
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        if self.output in ("degrees", "stats"):
            return None
        return self._graph.edges
    
    @property
    def n_nodes(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_nodes
    
    @property
    def n_edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.n_edges
    
    def degree_sequence(self):
        """Degree sequence"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def stats(self, include_triangles: bool = False) -> dict:
        """Summary statistics (memory efficient, no dense matrix)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.summary(include_triangles=include_triangles)
    
    def adjacency_matrix(self, format: str = "sparse"):
        """Adjacency matrix (sparse by default)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(format=format)
    
    def edges_coo(self):
        """Return edges in COO format (src, dst, weight arrays)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.edges_coo()
    
    def as_networkx(self, force: bool = False):
        """Convert to NetworkX (refuses for n > 200k unless force=True)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx(force=force)


# Factory function
def build_network(x: NDArray[np.float64], method: str, **kwargs):
    """
    Build network from time series (factory pattern).
    
    Parameters
    ----------
    x : array
        Time series
    method : str
        'hvg', 'nvg', 'recurrence', 'transition'
    **kwargs
        Method-specific parameters
    
    Returns
    -------
    graph : Graph-like object
        Built network with .edges, .degree_sequence(), etc.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> hvg = build_network(x, 'hvg')
    >>> rn = build_network(x, 'recurrence', m=3, rule='knn', k=5)
    """
    builders = {
        'hvg': HVG,
        'nvg': NVG,
        'recurrence': RecurrenceNetwork,
        'transition': TransitionNetwork,
    }
    
    method = method.lower()
    if method not in builders:
        raise ValueError(f"Unknown method: {method}. Choose from {list(builders.keys())}")
    
    builder_cls = builders[method]
    builder = builder_cls(**kwargs)
    return builder.build(x)

