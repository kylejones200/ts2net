"""
Natural Visibility Graph (NVG) implementation.

This module provides the NVG class for converting time series to graphs
using the natural visibility algorithm.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import networkx as nx
from scipy import sparse


class SKMixin:
    """Mixin for scikit-learn compatibility."""
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not k.endswith('_')}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

# Try to import numba for acceleration
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        """Dummy decorator when numba is not available."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@njit(cache=True)
def _nvg_edges_numba(x: np.ndarray, weighted: bool = False, limit: int = -1, 
                     max_edges: int = -1, max_edges_per_node: int = -1):
    """
    Build NVG edges using sweepline (no dense matrix) with bounded work.
    
    Tie-Breaking Rule:
    Two points i and j are naturally visible if the line connecting (i, x[i])
    to (j, x[j]) does not intersect any intermediate points. If an intermediate
    point k lies exactly on the line (x[k] == line_height), visibility is blocked.
    Points below the line do not block visibility.
    
    Args:
        x: 1D array of time series values
        weighted: If True, weight edges by absolute difference
        limit: Maximum temporal distance (horizon limit, -1 = no limit)
        max_edges: Maximum total edges (-1 = no limit, critical scale control)
        max_edges_per_node: Maximum edges per node (-1 = no limit)
        
    Returns:
        rows, cols, weights, hit_limit: Edge arrays and flag if limit was hit
    """
    n = len(x)
    
    # Determine allocation size with caps
    if max_edges > 0:
        alloc_size = min(max_edges, n * (n - 1) // 2)
    elif limit > 0:
        alloc_size = min(n * limit, n * (n - 1) // 2)
    else:
        alloc_size = n * (n - 1) // 2  # Worst case
    
    rows = np.zeros(alloc_size, dtype=np.int64)
    cols = np.zeros(alloc_size, dtype=np.int64)
    if weighted:
        weights = np.zeros(alloc_size, dtype=np.float64)
    else:
        weights = None
    
    edge_count = 0
    edges_per_node = np.zeros(n, dtype=np.int64)
    hit_limit = False
    
    for i in range(n):
        for j in range(i + 1, n):
            # Apply horizon limit (critical scale control)
            if limit > 0 and (j - i) > limit:
                continue
            
            # Check per-node edge limit
            if max_edges_per_node > 0 and edges_per_node[i] >= max_edges_per_node:
                continue
            
            # Check total edge limit
            if max_edges > 0 and edge_count >= max_edges:
                hit_limit = True
                break
            
            # Check visibility
            visible = True
            for k in range(i + 1, j):
                y_line = x[i] + (x[j] - x[i]) * (k - i) / (j - i)
                if x[k] > y_line:
                    visible = False
                    break
            
            if visible:
                if edge_count >= alloc_size:
                    # Hit allocation limit - resize or stop
                    if max_edges > 0:
                        hit_limit = True
                        break
                    # Resize (should be rare with proper limits)
                    new_max = alloc_size * 2
                    new_rows = np.zeros(new_max, dtype=np.int64)
                    new_cols = np.zeros(new_max, dtype=np.int64)
                    new_rows[:alloc_size] = rows
                    new_cols[:alloc_size] = cols
                    rows, cols = new_rows, new_cols
                    if weighted:
                        new_weights = np.zeros(new_max, dtype=np.float64)
                        new_weights[:alloc_size] = weights
                        weights = new_weights
                    alloc_size = new_max
                
                rows[edge_count] = i
                cols[edge_count] = j
                if weighted:
                    weights[edge_count] = abs(x[i] - x[j])
                edge_count += 1
                edges_per_node[i] += 1
        
        if hit_limit:
            break
    
    # Trim to actual size
    rows = rows[:edge_count]
    cols = cols[:edge_count]
    if weighted and weights is not None:
        weights = weights[:edge_count]
    
    return rows, cols, weights, hit_limit


class NVG(SKMixin):
    """Natural Visibility Graph (NVG) for time series analysis.
    
    The NVG is constructed by connecting each point to all other points
    that are visible to it, where visibility is defined by a straight line
    that doesn't intersect any intermediate points.
    """
    
    def __init__(self, weighted: bool = False, sparse: bool = False, limit: Optional[int] = None,
                 max_edges: Optional[int] = None, max_edges_per_node: Optional[int] = None,
                 max_memory_mb: Optional[float] = None):
        """Initialize the NVG converter.
        
        Args:
            weighted: If True, edges will be weighted by the absolute difference
                between the connected points.
            sparse: If True, use sparse matrices for the adjacency matrix.
            limit: If provided, only connect points within temporal distance ≤ limit.
                   (R ts2net compatibility parameter, critical scale control)
            max_edges: Maximum total edges (safety cap, prevents memory blowup)
            max_edges_per_node: Maximum edges per node (additional scale control)
            max_memory_mb: Maximum memory in MB (converted to max_edges estimate)
        """
        self.weighted = weighted
        self.sparse = sparse
        self.limit = limit
        self.max_edges = max_edges
        self.max_edges_per_node = max_edges_per_node
        
        # Convert memory limit to edge limit (rough estimate: 16 bytes per edge)
        if max_memory_mb is not None:
            estimated_max_edges = int(max_memory_mb * 1024 * 1024 / 16)
            if self.max_edges is None or estimated_max_edges < self.max_edges:
                self.max_edges = estimated_max_edges
    
    def fit(self, x: np.ndarray) -> 'NVG':
        """Fit the NVG model to the input time series.
        
        Args:
            x: 1D array-like, shape (n_samples,)
                The input time series.
                
        Returns:
            self: Returns the instance itself.
        """
        self.x_ = np.asarray(x).squeeze()
        if self.x_.ndim != 1:
            raise ValueError("Input must be a 1D array")
        return self
    
    def transform(self) -> Tuple[nx.Graph, Union[sparse.csr_matrix, np.ndarray]]:
        """Transform the time series into an NVG.
        
        Returns:
            Tuple containing:
                - G: The natural visibility graph as a networkx Graph
                - A: The adjacency matrix as a sparse or dense array
        """
        if not hasattr(self, 'x_'):
            raise RuntimeError("Must call fit() before transform()")
            
        n = len(self.x_)
        
        # Apply default horizon limit for large series (scale control)
        # For n > 10k, default limit to 5000 to prevent memory blowup
        effective_limit = self.limit
        if effective_limit is None and n > 10_000:
            effective_limit = 5000  # Default horizon for large series
            import warnings
            warnings.warn(
                f"NVG: n={n} is large. Applying default horizon limit={effective_limit} "
                f"to prevent memory blowup. Set limit explicitly to override.",
                UserWarning
            )
        
        # Use fast Numba implementation if available (builds edges, not dense matrix)
        if HAS_NUMBA:
            limit_val = effective_limit if effective_limit is not None else -1
            max_edges_val = self.max_edges if self.max_edges is not None else -1
            max_per_node = self.max_edges_per_node if self.max_edges_per_node is not None else -1
            rows, cols, weights, hit_limit = _nvg_edges_numba(
                self.x_, self.weighted, limit_val, max_edges_val, max_per_node
            )
            
            if hit_limit:
                import warnings
                warnings.warn(
                    f"NVG: Hit edge limit (max_edges={self.max_edges}, "
                    f"max_edges_per_node={self.max_edges_per_node}). "
                    f"Graph is partial. Increase limits for full graph.",
                    UserWarning
                )
            
            # Build NetworkX graph from edges
            G = nx.Graph()
            G.add_nodes_from(range(n))
            
            if self.weighted and weights is not None:
                for i in range(len(rows)):
                    G.add_edge(int(rows[i]), int(cols[i]), weight=float(weights[i]))
            else:
                for i in range(len(rows)):
                    G.add_edge(int(rows[i]), int(cols[i]))
            
            # Build sparse matrix from edges (never dense)
            if weights is not None:
                A = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
                # Make symmetric for undirected graph
                A = A + A.T
            else:
                data = np.ones(len(rows), dtype=float)
                A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
                A = A + A.T
            
            return G, A
        
        # Fallback to original implementation if Numba not available
        # Always use sparse to avoid memory blowup
        
        # Apply default horizon limit for large series
        effective_limit = self.limit
        if effective_limit is None and n > 10_000:
            effective_limit = 5000
            import warnings
            warnings.warn(
                f"NVG: n={n} is large. Applying default horizon limit={effective_limit}. "
                f"Set limit explicitly to override.",
                UserWarning
            )
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        rows, cols, data = [], [], []
        
        # Build the NVG by checking natural visibility between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Apply horizon limit (critical scale control)
                if effective_limit is not None and (j - i) > effective_limit:
                    continue
                
                # Check if there's a clear line of sight between i and j
                visible = True
                for k in range(i + 1, j):
                    # Check if point k is above the line connecting i and j
                    if self.x_[k] > self.x_[i] + (self.x_[j] - self.x_[i]) * (k - i) / (j - i):
                        visible = False
                        break
                
                if visible:
                    G.add_edge(i, j)
                    weight = abs(self.x_[i] - self.x_[j]) if self.weighted else 1.0
                    rows.append(i)
                    cols.append(j)
                    data.append(weight)
        
        # Build sparse matrix (never dense)
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        # Make symmetric for undirected graph
        A = A + A.T
        
        return G, A
    
    def fit_transform(self, x: np.ndarray) -> Tuple[nx.Graph, Union[sparse.csr_matrix, np.ndarray]]:
        """Fit the model to the input time series and transform it to an NVG.
        
        Args:
            x: 1D array-like, shape (n_samples,)
                The input time series.
                
        Returns:
            Tuple containing:
                - G: The natural visibility graph as a networkx Graph
                - A: The adjacency matrix as a sparse or dense array
        """
        return self.fit(x).transform()
