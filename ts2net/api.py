"""
Wraps existing implementations with new lightweight Graph interface.
All old functionality preserved via .fit_transform().
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union, Literal, List
import networkx as nx
from .core.graph import Graph
from .core.visibility.weights import compute_weight, WeightMode

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
    
    # Enhanced validation with warnings for common issues
    import warnings
    
    # Check for constant series (can cause issues in some methods)
    if np.std(x) == 0:
        warnings.warn(
            f"{method_name}: Constant series detected (std=0). "
            f"Results may be degenerate.",
            UserWarning
        )
    
    # Check for very short series
    if len(x) < 3:
        warnings.warn(
            f"{method_name}: Very short series (n={len(x)}). "
            f"Network may be trivial or degenerate. Consider using longer series.",
            UserWarning
        )
    
    # Check for very long series (may be slow)
    if len(x) > 100_000:
        warnings.warn(
            f"{method_name}: Very long series (n={len(x)}). "
            f"This may be slow. Consider using limit parameter or resampling.",
            UserWarning
        )
    
    # Check for potential numerical issues
    if np.any(np.abs(x) > 1e10):
        warnings.warn(
            f"{method_name}: Series contains very large values (max={np.max(np.abs(x)):.2e}). "
            f"This may cause numerical issues.",
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
    
    def __init__(self, weighted: Union[bool, str] = False, limit: Optional[int] = None, 
                 only_degrees: bool = False, output: str = "edges", directed: bool = False,
                 weight_mode: Optional[str] = None):
        """
        Parameters
        ----------
        weighted : bool or str, default False
            If True, use "absdiff" weight mode. If str, use that weight mode.
            Valid modes: "absdiff", "time_gap", "slope", "min_clearance"
        weight_mode : str, optional
            Explicit weight mode (overrides weighted if provided).
            Valid modes: "absdiff", "time_gap", "slope", "min_clearance"
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
        # Resolve weight mode
        if weight_mode is not None:
            self.weight_mode = weight_mode
            self.weighted = True
        elif isinstance(weighted, str):
            self.weight_mode = weighted
            self.weighted = True
        elif weighted is True:
            self.weight_mode = "absdiff"  # Default mode
            self.weighted = True
        else:
            self.weight_mode = None
            self.weighted = False
        self.limit = limit
        self.directed = directed
        # Backward compatibility
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        # For _HVG_Old, use bool weighted (it only supports absdiff)
        # We'll recompute weights after if needed
        self._impl = _HVG_Old(weighted=(self.weighted and self.weight_mode == "absdiff"), 
                             limit=limit, directed=directed)
        self._graph = None
        self._x = None  # Store series for weight recomputation
    
    def fit(self, x: NDArray[np.float64]) -> 'HVG':
        """
        Fit the HVG model to the time series (scikit-learn compatible).
        
        Parameters
        ----------
        x : array-like
            Input time series (1D array)
        
        Returns
        -------
        self : HVG
            Returns self for method chaining
        
        Examples
        --------
        >>> from ts2net import HVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> hvg = HVG()
        >>> hvg.fit(x)  # scikit-learn style
        >>> hvg.transform()  # Get graph
        """
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "HVG")
        self._x = x.copy()  # Store for weight recomputation
        self._fitted = True
        return self
    
    def transform(self) -> nx.Graph:
        """
        Transform the fitted time series into a network (scikit-learn compatible).
        
        Returns
        -------
        G : networkx.Graph or DiGraph
            The visibility graph
        
        Raises
        ------
        ValueError
            If fit() has not been called first
        
        Examples
        --------
        >>> hvg = HVG()
        >>> hvg.fit(x)
        >>> G = hvg.transform()
        """
        if not getattr(self, '_fitted', False):
            raise ValueError("Must call fit() before transform()")
        if self._graph is None:
            self.build(self._x)
        return self.as_networkx(force=True)
    
    def fit_transform(self, x: NDArray[np.float64]) -> nx.Graph:
        """
        Fit the model and transform in one step (scikit-learn compatible).
        
        Parameters
        ----------
        x : array-like
            Input time series
        
        Returns
        -------
        G : networkx.Graph or DiGraph
            The visibility graph
        
        Examples
        --------
        >>> hvg = HVG()
        >>> G = hvg.fit_transform(x)
        """
        return self.fit(x).transform()
    
    def _compute_degrees(self, G_nx: nx.Graph) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        """Compute degree sequences (vectorized)."""
        if self.directed:
            out_degrees = np.array([d for _, d in G_nx.out_degree()])
            in_degrees = np.array([d for _, d in G_nx.in_degree()])
            return out_degrees, in_degrees, out_degrees
        degrees = np.array([d for _, d in G_nx.degree()])
        return degrees, None, None
    
    def _build_degrees_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in degrees-only mode."""
        degrees, in_degrees, out_degrees = self._compute_degrees(G_nx)
        return Graph(
            edges=[],
            n_nodes=len(x),
            directed=self.directed,
            weighted=self.weighted,
            _adjacency=None,
            _degrees=degrees,
            _in_degrees=in_degrees,
            _out_degrees=out_degrees
        )
    
    def _build_stats_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in stats-only mode."""
        degrees, in_degrees, out_degrees = self._compute_degrees(G_nx)
        graph = Graph(
            edges=[],
            n_nodes=len(x),
            directed=self.directed,
            weighted=self.weighted,
            _adjacency=None,
            _degrees=degrees,
            _in_degrees=in_degrees,
            _out_degrees=out_degrees
        )
        graph._n_edges_cached = G_nx.number_of_edges()
        return graph
    
    def _build_edges_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in full edges mode."""
        edges = list(G_nx.edges(data='weight' if self.weighted else False))
        
        if self.weighted and self.weight_mode and self.weight_mode != "absdiff":
            edge_pairs = [(u, v) for u, v, _ in edges]
            edges = [(u, v, compute_weight(x, u, v, self.weight_mode)) for u, v in edge_pairs]
        elif self.weighted:
            edges = [(u, v, w) for u, v, w in edges]
        else:
            edges = [(u, v) for u, v in edges]
        
        return Graph(
            edges=edges,
            n_nodes=len(x),
            directed=self.directed,
            weighted=self.weighted,
            _adjacency=None
        )
    
    def build(self, x: NDArray[np.float64]):
        """
        Build HVG from time series (legacy method, use fit() for scikit-learn compatibility).
        
        .. deprecated:: Use fit() and transform() for scikit-learn compatibility.
        
        Parameters
        ----------
        x : array-like
            Input time series
        
        Returns
        -------
        self : HVG
            Returns self for method chaining
        """
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "HVG")
        self._x = x.copy()  # Store for weight recomputation
        
        # Use old implementation
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert to new Graph object based on output mode
        output_handlers = {
            "degrees": lambda: self._build_degrees_graph(G_nx, x),
            "stats": lambda: self._build_stats_graph(G_nx, x),
            "edges": lambda: self._build_edges_graph(G_nx, x),
        }
        
        handler = output_handlers.get(self.output)
        if handler is None:
            raise ValueError(f"Unknown output mode: {self.output}")
        
        self._graph = handler()
        
        self._fitted = True
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
            Additional arguments passed to metric functions (e.g., method, weight, resolution, seed)
        
        Returns
        -------
        dict
            Dictionary with network metrics:
            - Clustering: avg_clustering, transitivity
            - Path lengths: avg_path_length, diameter, radius
            - Modularity: modularity, n_communities
        
        Examples
        --------
        >>> from ts2net import HVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> hvg = HVG()
        >>> hvg.build(x)
        >>> metrics = hvg.network_metrics()
        >>> print(f"Clustering: {metrics['avg_clustering']:.3f}")
        >>> print(f"Avg path length: {metrics['avg_path_length']:.3f}")
        >>> print(f"Modularity: {metrics['modularity']:.3f}")
        """
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.network_metrics(include=include, sample_size=sample_size, **kwargs)
    
    def test_significance(
        self,
        metric: str = "density",
        method: str = "shuffle",
        n_surrogates: int = 200,
        alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
        **kwargs
    ):
        """
        Test significance of a network metric against null distribution.
        
        Parameters
        ----------
        metric : str, default "density"
            Metric to test. Options: "density", "deg_mean", "deg_std", 
            "avg_clustering", "assortativity", or any key from stats()
        method : str, default "shuffle"
            Surrogate generation method: "shuffle", "phase", "circular", "iaaft", "block_bootstrap"
        n_surrogates : int, default 200
            Number of surrogate series to generate
        alpha : float, default 0.05
            Significance level (two-tailed)
        rng : np.random.Generator, optional
            Random number generator
        **kwargs
            Additional arguments for surrogate generation (e.g., block_size for block_bootstrap)
        
        Returns
        -------
        result : NetworkSignificanceResult
            Significance test result
        
        Examples
        --------
        >>> from ts2net import HVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> hvg = HVG()
        >>> hvg.build(x)
        >>> result = hvg.test_significance(metric="density", method="phase", n_surrogates=100)
        >>> print(result)
        """
        from .stats.null_models import compute_network_metric_significance
        
        if self._graph is None:
            raise ValueError("Call build() first")
        
        if self._x is None:
            raise ValueError("Cannot test significance: original time series not stored")
        
        # Create metric function
        def metric_fn(ts):
            hvg_temp = HVG(
                weighted=self.weighted,
                weight_mode=self.weight_mode,
                limit=self.limit,
                directed=self.directed,
                output=self.output
            )
            hvg_temp.build(ts)
            stats = hvg_temp.stats()
            if metric not in stats:
                raise ValueError(f"Unknown metric: {metric}. Available: {list(stats.keys())}")
            return float(stats[metric])
        
        return compute_network_metric_significance(
            self._x,
            metric_fn,
            method=method,
            n_surrogates=n_surrogates,
            alpha=alpha,
            metric_name=metric,
            rng=rng,
            **kwargs
        )
    
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
    
    def __init__(self, weighted: Union[bool, str] = False, limit: Optional[int] = None,
                 only_degrees: bool = False, output: str = "edges",
                 max_edges: Optional[int] = None, max_edges_per_node: Optional[int] = None,
                 max_memory_mb: Optional[float] = None, weight_mode: Optional[str] = None):
        """
        Parameters
        ----------
        weighted : bool or str, default False
            If True, use "absdiff" weight mode. If str, use that weight mode.
            Valid modes: "absdiff", "time_gap", "slope", "min_clearance"
        weight_mode : str, optional
            Explicit weight mode (overrides weighted if provided).
            Valid modes: "absdiff", "time_gap", "slope", "min_clearance"
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
        # Resolve weight mode (same logic as HVG)
        if weight_mode is not None:
            self.weight_mode = weight_mode
            self.weighted = True
        elif isinstance(weighted, str):
            self.weight_mode = weighted
            self.weighted = True
        elif weighted is True:
            self.weight_mode = "absdiff"  # Default mode
            self.weighted = True
        else:
            self.weight_mode = None
            self.weighted = False
        
        self.limit = limit
        if only_degrees:
            self.output = "degrees"
        else:
            self.output = output
        # For _NVG_Old, use bool weighted (it only supports absdiff)
        self._impl = _NVG_Old(
            weighted=(self.weighted and self.weight_mode == "absdiff"), limit=limit,
            max_edges=max_edges, max_edges_per_node=max_edges_per_node,
            max_memory_mb=max_memory_mb
        )
        self._graph = None
        self._x = None  # Store series for weight recomputation
    
    def fit(self, x: NDArray[np.float64]) -> 'NVG':
        """
        Fit the NVG model to the time series (scikit-learn compatible).
        
        Parameters
        ----------
        x : array-like
            Input time series (1D array)
        
        Returns
        -------
        self : NVG
            Returns self for method chaining
        
        Examples
        --------
        >>> from ts2net import NVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> nvg = NVG()
        >>> nvg.fit(x)  # scikit-learn style
        >>> nvg.transform()  # Get graph
        """
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "NVG")
        self._x = x.copy()  # Store for weight recomputation
        self._fitted = True
        return self
    
    def transform(self) -> nx.Graph:
        """
        Transform the fitted time series into a network (scikit-learn compatible).
        
        Returns
        -------
        G : networkx.Graph
            The visibility graph
        
        Raises
        ------
        ValueError
            If fit() has not been called first
        
        Examples
        --------
        >>> nvg = NVG()
        >>> nvg.fit(x)
        >>> G = nvg.transform()
        """
        if not getattr(self, '_fitted', False):
            raise ValueError("Must call fit() before transform()")
        if self._graph is None:
            self.build(self._x)
        return self.as_networkx(force=True)
    
    def fit_transform(self, x: NDArray[np.float64]) -> nx.Graph:
        """
        Fit the model and transform in one step (scikit-learn compatible).
        
        Parameters
        ----------
        x : array-like
            Input time series
        
        Returns
        -------
        G : networkx.Graph
            The visibility graph
        
        Examples
        --------
        >>> nvg = NVG()
        >>> G = nvg.fit_transform(x)
        """
        return self.fit(x).transform()
    
    def build(self, x: NDArray[np.float64]):
        """
        Build NVG from time series (legacy method, use fit() for scikit-learn compatibility).
        
        .. deprecated:: Use fit() and transform() for scikit-learn compatibility.
        
        Parameters
        ----------
        x : array-like
            Input time series
        
        Returns
        -------
        self : NVG
            Returns self for method chaining
        """
        # Validate and clean input (handles dtype contamination)
        x = _validate_and_clean_series(x, "NVG")
        self._x = x.copy()  # Store for weight recomputation
        
        G_nx, A = self._impl.fit_transform(x)
        
        output_handlers = {
            "degrees": lambda: self._build_degrees_graph(G_nx, x),
            "stats": lambda: self._build_stats_graph(G_nx, x),
            "edges": lambda: self._build_edges_graph(G_nx, x),
        }
        
        handler = output_handlers.get(self.output)
        if handler is None:
            raise ValueError(f"Unknown output mode: {self.output}")
        
        self._graph = handler()
        self._fitted = True
        return self
    
    def _build_degrees_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in degrees-only mode."""
        degrees = np.array([d for _, d in G_nx.degree()])
        return Graph(
            edges=[],
            n_nodes=len(x),
            directed=False,
            weighted=self.weighted,
            _adjacency=None,
            _degrees=degrees
        )
    
    def _build_stats_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in stats-only mode."""
        degrees = np.array([d for _, d in G_nx.degree()])
        graph = Graph(
            edges=[],
            n_nodes=len(x),
            directed=False,
            weighted=self.weighted,
            _adjacency=None,
            _degrees=degrees
        )
        graph._n_edges_cached = G_nx.number_of_edges()
        return graph
    
    def _build_edges_graph(self, G_nx: nx.Graph, x: NDArray) -> Graph:
        """Build graph in full edges mode."""
        edges = list(G_nx.edges(data='weight' if self.weighted else False))
        
        if self.weighted and self.weight_mode and self.weight_mode != "absdiff":
            edge_pairs = [(u, v) for u, v, _ in edges]
            edges = [(u, v, compute_weight(x, u, v, self.weight_mode)) for u, v in edge_pairs]
        elif self.weighted:
            edges = [(u, v, w) for u, v, w in edges]
        else:
            edges = [(u, v) for u, v in edges]
        
        return Graph(
            edges=edges,
            n_nodes=len(x),
            directed=False,
            weighted=self.weighted,
            _adjacency=None
        )
    
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
            Additional arguments passed to metric functions (e.g., method, weight, resolution, seed)
        
        Returns
        -------
        dict
            Dictionary with network metrics:
            - Clustering: avg_clustering, transitivity
            - Path lengths: avg_path_length, diameter, radius
            - Modularity: modularity, n_communities
        
        Examples
        --------
        >>> from ts2net import HVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> hvg = HVG()
        >>> hvg.build(x)
        >>> metrics = hvg.network_metrics()
        >>> print(f"Clustering: {metrics['avg_clustering']:.3f}")
        >>> print(f"Avg path length: {metrics['avg_path_length']:.3f}")
        >>> print(f"Modularity: {metrics['modularity']:.3f}")
        """
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.network_metrics(include=include, sample_size=sample_size, **kwargs)
    
    def test_significance(
        self,
        metric: str = "density",
        method: str = "shuffle",
        n_surrogates: int = 200,
        alpha: float = 0.05,
        rng: Optional[np.random.Generator] = None,
        **kwargs
    ):
        """
        Test significance of a network metric against null distribution.
        
        Parameters
        ----------
        metric : str, default "density"
            Metric to test. Options: "density", "deg_mean", "deg_std", 
            "avg_clustering", "assortativity", or any key from stats()
        method : str, default "shuffle"
            Surrogate generation method: "shuffle", "phase", "circular", "iaaft", "block_bootstrap"
        n_surrogates : int, default 200
            Number of surrogate series to generate
        alpha : float, default 0.05
            Significance level (two-tailed)
        rng : np.random.Generator, optional
            Random number generator
        **kwargs
            Additional arguments for surrogate generation (e.g., block_size for block_bootstrap)
        
        Returns
        -------
        result : NetworkSignificanceResult
            Significance test result
        
        Examples
        --------
        >>> from ts2net import NVG
        >>> import numpy as np
        >>> x = np.random.randn(100)
        >>> nvg = NVG()
        >>> nvg.build(x)
        >>> result = nvg.test_significance(metric="density", method="phase", n_surrogates=100)
        >>> print(result)
        """
        from .stats.null_models import compute_network_metric_significance
        
        if self._graph is None:
            raise ValueError("Call build() first")
        
        if self._x is None:
            raise ValueError("Cannot test significance: original time series not stored")
        
        # Create metric function
        def metric_fn(ts):
            # Get parameters from implementation, with safe defaults
            max_edges = getattr(self._impl, 'max_edges', None)
            max_edges_per_node = getattr(self._impl, 'max_edges_per_node', None)
            max_memory_mb = getattr(self._impl, 'max_memory_mb', None)
            
            nvg_temp = NVG(
                weighted=self.weighted,
                weight_mode=self.weight_mode,
                limit=self.limit,
                max_edges=max_edges,
                max_edges_per_node=max_edges_per_node,
                max_memory_mb=max_memory_mb,
                output=self.output
            )
            nvg_temp.build(ts)
            stats = nvg_temp.stats()
            if metric not in stats:
                raise ValueError(f"Unknown metric: {metric}. Available: {list(stats.keys())}")
            return float(stats[metric])
        
        return compute_network_metric_significance(
            self._x,
            metric_fn,
            method=method,
            n_surrogates=n_surrogates,
            alpha=alpha,
            metric_name=metric,
            rng=rng,
            **kwargs
        )
    
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

