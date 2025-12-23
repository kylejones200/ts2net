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


def _validate_and_clean_series(x, method_name: str = "ts2net") -> NDArray[np.float64]:
    """
    Validate and clean time series input, handling dtype contamination.
    
    Parameters
    ----------
    x : array-like
        Input time series (may contain non-numeric values)
    method_name : str
        Method name for error messages
    
    Returns
    -------
    x : array (float64)
        Clean numeric array
    
    Raises
    ------
    ValueError
        If input cannot be converted to valid numeric array
    """
    import pandas as pd
    
    # Convert to numpy array first
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # Check if already numeric
    if x.dtype.kind in ['f', 'i', 'u']:
        # Numeric type - just ensure float64 and check for inf/nan
        x = x.astype(np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        x = x[~np.isnan(x)]
    else:
        # Non-numeric or object dtype - use pandas coercion
        s = pd.Series(x)
        s = pd.to_numeric(s, errors='coerce')
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.dropna()
        x = s.values.astype(np.float64)
    
    # Final validation
    if len(x) == 0:
        raise ValueError(
            f"{method_name}: No valid numeric values in input series. "
            f"Check for non-numeric data, infinities, or all-null values."
        )
    
    if x.ndim != 1:
        raise ValueError(
            f"{method_name}: Input must be 1D array, got shape {x.shape}"
        )
    
    # Check for constant series (can cause issues in some methods)
    if np.std(x) == 0:
        import warnings
        warnings.warn(
            f"{method_name}: Constant series detected (std=0). "
            f"Results may be degenerate.",
            UserWarning
        )
    
    return x


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
                 only_degrees: bool = False, output: str = "edges", directed: bool = False):
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
        directed : bool, default False
            If True, create directed graph with edges forward in time (i → j where i < j).
            Enables irreversibility analysis useful for fault detection in time series.
        """
        self.weighted = weighted
        self.limit = limit
        self.directed = directed
        # Backward compatibility
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        self._impl = _HVG_Old(weighted=weighted, limit=limit, directed=directed)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build HVG from time series."""
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "HVG")
        
        # Use old implementation
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert to new Graph object based on output mode
        if self.output == "degrees":
            # Degrees-only mode: compute degrees, skip edges
            if self.directed:
                # For directed graphs, compute both in and out degrees
                out_degrees = np.array([d for _, d in G_nx.out_degree()])
                in_degrees = np.array([d for _, d in G_nx.in_degree()])
                # For backward compatibility, use out_degrees as primary
                degrees = out_degrees
            else:
                degrees = np.array([d for _, d in G_nx.degree()])
                in_degrees = None
                out_degrees = None
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=self.directed,
                weighted=self.weighted,
                _adjacency=None,  # Don't store sparse matrix either
                _degrees=degrees,
                _in_degrees=in_degrees,
                _out_degrees=out_degrees
            )
        elif self.output == "stats":
            # Stats-only mode: compute degrees and basic stats, skip edges
            if self.directed:
                out_degrees = np.array([d for _, d in G_nx.out_degree()])
                in_degrees = np.array([d for _, d in G_nx.in_degree()])
                # For backward compatibility, use out_degrees as primary
                degrees = out_degrees
            else:
                degrees = np.array([d for _, d in G_nx.degree()])
                in_degrees = None
                out_degrees = None
            n_edges = G_nx.number_of_edges()
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=self.directed,
                weighted=self.weighted,
                _adjacency=None,
                _degrees=degrees,
                _in_degrees=in_degrees,
                _out_degrees=out_degrees
            )
            # Store edge count separately since we don't have edges
            self._graph._n_edges_cached = n_edges
        else:  # output == "edges"
            # Full edge list mode
            edges = list(G_nx.edges(data='weight' if self.weighted else False))
            if self.weighted:
                edges = [(u, v, w) for u, v, w in edges]
            else:
                edges = [(u, v) for u, v in edges]
            
            self._graph = Graph(
                edges=edges,
                n_nodes=len(x),
                directed=self.directed,
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
        """Degree sequence (out-degree for directed graphs, total degree for undirected)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def in_degree_sequence(self):
        """In-degree sequence (only valid for directed graphs)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        if not self.directed:
            raise ValueError("in_degree_sequence() only valid for directed graphs")
        return self._graph.in_degree_sequence()
    
    def out_degree_sequence(self):
        """Out-degree sequence (only valid for directed graphs)"""
        if self._graph is None:
            raise ValueError("Call build() first")
        if not self.directed:
            raise ValueError("out_degree_sequence() only valid for directed graphs")
        return self._graph.out_degree_sequence()
    
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
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "NVG")
        
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
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "RecurrenceNetwork")
        
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
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "TransitionNetwork")
        
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

