"""
Horizontal Visibility Graph (HVG) implementation.

This module provides the HVG class for converting time series to graphs
using the horizontal visibility algorithm.
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
def _hvg_edges_numba(x: np.ndarray, weighted: bool = False, limit: int = -1):
    """
    Build HVG edges using O(n) stack algorithm (no dense matrix).
    
    Uses a stack-based algorithm for O(n) expected time complexity.
    Returns edge list to avoid building dense adjacency matrix.
    
    Tie-Breaking Rule:
    Two points i and j are horizontally visible if all points k between them
    satisfy x[k] < min(x[i], x[j]). If any intermediate point equals or exceeds
    min(x[i], x[j]), visibility is blocked. This means:
    - Points with the same value are visible only if all intermediate points
      are strictly less than that value
    - Consecutive points are always visible (no intermediate points to block)
    
    Args:
        x: 1D array of time series values
        weighted: If True, weight edges by absolute difference
        limit: Maximum temporal distance (-1 = no limit)
        
    Returns:
        edges: List of (i, j) or (i, j, weight) tuples
    """
    n = len(x)
    # Pre-allocate lists (will resize as needed, but this is faster than appending)
    max_edges = 2 * n  # HVG typically has ~2n edges
    rows = np.zeros(max_edges, dtype=np.int64)
    cols = np.zeros(max_edges, dtype=np.int64)
    if weighted:
        weights = np.zeros(max_edges, dtype=np.float64)
    else:
        weights = None
    
    edge_count = 0
    stack = []
    
    for j in range(n):
        # Pop all points that are smaller than current point
        while len(stack) > 0 and x[stack[-1]] < x[j]:
            i = stack.pop()
            # Check limit
            if limit > 0 and abs(j - i) > limit:
                continue
            # Add edge
            if edge_count >= max_edges:
                # Resize (rare case)
                new_max = max_edges * 2
                new_rows = np.zeros(new_max, dtype=np.int64)
                new_cols = np.zeros(new_max, dtype=np.int64)
                new_rows[:max_edges] = rows
                new_cols[:max_edges] = cols
                rows, cols = new_rows, new_cols
                if weighted:
                    new_weights = np.zeros(new_max, dtype=np.float64)
                    new_weights[:max_edges] = weights
                    weights = new_weights
                max_edges = new_max
            
            rows[edge_count] = i
            cols[edge_count] = j
            if weighted:
                weights[edge_count] = abs(x[i] - x[j])
            edge_count += 1
        
        # The top of the stack (if exists) can also see current point
        if len(stack) > 0:
            i = stack[-1]
            if limit <= 0 or abs(j - i) <= limit:
                if edge_count >= max_edges:
                    new_max = max_edges * 2
                    new_rows = np.zeros(new_max, dtype=np.int64)
                    new_cols = np.zeros(new_max, dtype=np.int64)
                    new_rows[:max_edges] = rows
                    new_cols[:max_edges] = cols
                    rows, cols = new_rows, new_cols
                    if weighted:
                        new_weights = np.zeros(new_max, dtype=np.float64)
                        new_weights[:max_edges] = weights
                        weights = new_weights
                    max_edges = new_max
                
                rows[edge_count] = i
                cols[edge_count] = j
                if weighted:
                    weights[edge_count] = abs(x[i] - x[j])
                edge_count += 1
        
        stack.append(j)
    
    # Trim to actual size
    rows = rows[:edge_count]
    cols = cols[:edge_count]
    if weighted:
        weights = weights[:edge_count]
    
    return rows, cols, weights


class HVG(SKMixin):
    """Horizontal Visibility Graph (HVG) for time series analysis.
    
    The HVG is constructed by connecting each point to its horizontal
    neighbors that can be seen by a horizontal line of sight.
    
    For directed graphs, edges point forward in time (i → j where i < j),
    enabling irreversibility analysis useful for fault detection.
    """
    
    def __init__(self, weighted: bool = False, sparse: bool = False, limit: Optional[int] = None,
                 directed: bool = False):
        """Initialize the HVG converter.
        
        Args:
            weighted: If True, edges will be weighted by the absolute difference
                between the connected points.
            sparse: If True, use sparse matrices for the adjacency matrix.
            limit: If provided, only connect points within temporal distance ≤ limit.
                   (R ts2net compatibility parameter)
            directed: If True, create directed graph with edges forward in time.
                     Enables irreversibility analysis for fault detection.
        """
        self.weighted = weighted
        self.sparse = sparse
        self.limit = limit
        self.directed = directed
    
    def fit(self, x: np.ndarray) -> 'HVG':
        """Fit the HVG model to the input time series.
        
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
        """Transform the time series into an HVG.
        
        Returns:
            Tuple containing:
                - G: The horizontal visibility graph as a networkx Graph
                - A: The adjacency matrix as a sparse or dense array
        """
        if not hasattr(self, 'x_'):
            raise RuntimeError("Must call fit() before transform()")
            
        n = len(self.x_)
        
        # Use fast Numba implementation if available (builds edges, not dense matrix)
        if HAS_NUMBA:
            limit_val = self.limit if self.limit is not None else -1
            rows, cols, weights = _hvg_edges_numba(self.x_, self.weighted, limit_val)
            
            # Build NetworkX graph from edges (directed or undirected)
            if self.directed:
                G = nx.DiGraph()
            else:
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
            else:
                data = np.ones(len(rows), dtype=float)
                A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
            
            # Make symmetric for undirected graph
            if not self.directed:
                A = A + A.T
            
            return G, A
        
        # Fallback to original implementation if Numba not available
        # Always use sparse to avoid memory blowup
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        G.add_nodes_from(range(n))
        
        rows, cols, data = [], [], []
        
        # Build the HVG by checking horizontal visibility between all pairs
        # For directed graphs, only create edges forward in time (i → j where i < j)
        for i in range(n):
            for j in range(i + 1, n):
                # Apply limit if specified
                if self.limit is not None and (j - i) > self.limit:
                    continue
                
                # Check if there's a clear line of sight between i and j
                visible = True
                for k in range(i + 1, j):
                    if self.x_[k] > min(self.x_[i], self.x_[j]):
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
        if not self.directed:
            A = A + A.T
        
        return G, A
    
    def fit_transform(self, x: np.ndarray) -> Tuple[nx.Graph, Union[sparse.csr_matrix, np.ndarray]]:
        """Fit the model to the input time series and transform it to an HVG.
        
        Args:
            x: 1D array-like, shape (n_samples,)
                The input time series.
                
        Returns:
            Tuple containing:
                - G: The horizontal visibility graph as a networkx Graph
                - A: The adjacency matrix as a sparse or dense array
        """
        return self.fit(x).transform()
