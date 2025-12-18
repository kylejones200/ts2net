"""
Wraps existing implementations with new lightweight Graph interface.
All old functionality preserved via .fit_transform().
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from .core.graph import Graph

# Import existing implementations from core/
import sys
import os
core_path = os.path.join(os.path.dirname(__file__), '..')
if core_path not in sys.path:
    sys.path.insert(0, core_path)

from core.visibility.hvg import HVG as _HVG_Old
from core.visibility.nvg import NVG as _NVG_Old
from core.recurrence import RecurrenceNetwork as _RN_Old
from core.transition import TransitionNetwork as _TN_Old


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
                 only_degrees: bool = False):
        self.weighted = weighted
        self.limit = limit
        self.only_degrees = only_degrees
        self._impl = _HVG_Old(weighted=weighted, limit=limit)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build HVG from time series."""
        # Use old implementation
        G_nx, A = self._impl.fit_transform(x)
        
        # Convert to new Graph object
        if self.only_degrees:
            # Performance mode
            degrees = np.array([d for _, d in G_nx.degree()])
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=A,
                _degrees=degrees
            )
        else:
            # Normal mode
            edges = list(G_nx.edges(data='weight' if self.weighted else False))
            if self.weighted:
                edges = [(u, v, w) for u, v, w in edges]
            
            self._graph = Graph(
                edges=edges,
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=A
            )
        
        return self
    
    @property
    def edges(self):
        """Edge list"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return None if self.only_degrees else self._graph.edges
    
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
    
    def adjacency_matrix(self, sparse: bool = False):
        """Adjacency matrix"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(sparse=sparse)
    
    def as_networkx(self):
        """Convert to NetworkX"""
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx()


class NVG:
    """Natural Visibility Graph"""
    
    def __init__(self, weighted: bool = False, limit: Optional[int] = None,
                 only_degrees: bool = False):
        self.weighted = weighted
        self.limit = limit
        self.only_degrees = only_degrees
        self._impl = _NVG_Old(weighted=weighted, limit=limit)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build NVG from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        if self.only_degrees:
            degrees = np.array([d for _, d in G_nx.degree()])
            self._graph = Graph(
                edges=[],
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=A,
                _degrees=degrees
            )
        else:
            edges = list(G_nx.edges(data='weight' if self.weighted else False))
            if self.weighted:
                edges = [(u, v, w) for u, v, w in edges]
            
            self._graph = Graph(
                edges=edges,
                n_nodes=len(x),
                directed=False,
                weighted=self.weighted,
                _adjacency=A
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return None if self.only_degrees else self._graph.edges
    
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
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def adjacency_matrix(self, sparse: bool = False):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(sparse=sparse)
    
    def as_networkx(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx()


class RecurrenceNetwork:
    """Recurrence Network"""
    
    def __init__(self, m: Optional[int] = None, tau: int = 1, rule: str = 'knn',
                 k: int = 5, epsilon: float = 0.1, metric: str = 'euclidean',
                 only_degrees: bool = False):
        self.m = m
        self.tau = tau
        self.rule = rule
        self.k = k
        self.epsilon = epsilon
        self.metric = metric
        self.only_degrees = only_degrees
        self._impl = _RN_Old(m=m, tau=tau, rule=rule, k=k, epsilon=epsilon, metric=metric)
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build recurrence network from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        if self.only_degrees:
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=False,
                weighted=False,
                _adjacency=A,
                _degrees=degrees
            )
        else:
            edges = list(G_nx.edges())
            self._graph = Graph(
                edges=edges,
                n_nodes=G_nx.number_of_nodes(),
                directed=False,
                weighted=False,
                _adjacency=A
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return None if self.only_degrees else self._graph.edges
    
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
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def adjacency_matrix(self, sparse: bool = False):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(sparse=sparse)
    
    def as_networkx(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx()


class TransitionNetwork:
    """Transition Network"""
    
    def __init__(self, symbolizer: str = 'ordinal', order: int = 3, delay: int = 1,
                 tie_rule: str = 'stable', bins: int = 5, normalize: bool = True,
                 sparse: bool = False, only_degrees: bool = False):
        self.symbolizer = symbolizer
        self.order = order
        self.delay = delay
        self.tie_rule = tie_rule
        self.bins = bins
        self.normalize = normalize
        self.sparse = sparse
        self.only_degrees = only_degrees
        self._impl = _TN_Old(
            symbolizer=symbolizer, order=order, delay=delay,
            tie_rule=tie_rule, bins=bins, normalize=normalize, sparse=sparse
        )
        self._graph = None
    
    def build(self, x: NDArray[np.float64]):
        """Build transition network from time series."""
        G_nx, A = self._impl.fit_transform(x)
        
        if self.only_degrees:
            degrees = np.array([d for _, d in G_nx.degree()])
            n = G_nx.number_of_nodes()
            self._graph = Graph(
                edges=[],
                n_nodes=n,
                directed=True,  # Transition networks are directed
                weighted=False,
                _adjacency=A,
                _degrees=degrees
            )
        else:
            edges = list(G_nx.edges())
            self._graph = Graph(
                edges=edges,
                n_nodes=G_nx.number_of_nodes(),
                directed=True,
                weighted=False,
                _adjacency=A
            )
        
        return self
    
    @property
    def edges(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return None if self.only_degrees else self._graph.edges
    
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
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.degree_sequence()
    
    def adjacency_matrix(self, sparse: bool = False):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.adjacency_matrix(sparse=sparse)
    
    def as_networkx(self):
        if self._graph is None:
            raise ValueError("Call build() first")
        return self._graph.as_networkx()


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

