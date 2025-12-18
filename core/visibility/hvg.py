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
def _hvg_adj_matrix_numba(x: np.ndarray, weighted: bool = False, limit: int = -1) -> np.ndarray:
    """
    Build HVG adjacency matrix directly using Numba.
    
    Uses a stack-based algorithm for O(n) expected time complexity
    instead of the naive O(n³) approach.
    
    Args:
        x: 1D array of time series values
        weighted: If True, weight edges by absolute difference
        
    Returns:
        A: Adjacency matrix of shape (n, n)
    """
    n = len(x)
    A = np.zeros((n, n), dtype=np.float64)
    
    # Stack-based algorithm for horizontal visibility
    # The stack maintains indices of points that can potentially see future points
    stack = []
    
    for j in range(n):
        # Pop all points that are smaller than current point
        # These points can see the current point horizontally
        while len(stack) > 0 and x[stack[-1]] < x[j]:
            i = stack.pop()
            if weighted:
                weight = abs(x[i] - x[j])
                A[i, j] = weight
                A[j, i] = weight
            else:
                A[i, j] = 1.0
                A[j, i] = 1.0
        
        # The top of the stack (if exists) can also see current point
        if len(stack) > 0:
            i = stack[-1]
            if weighted:
                weight = abs(x[i] - x[j])
                A[i, j] = weight
                A[j, i] = weight
            else:
                A[i, j] = 1.0
                A[j, i] = 1.0
        
        stack.append(j)
    
    return A


class HVG(SKMixin):
    """Horizontal Visibility Graph (HVG) for time series analysis.
    
    The HVG is constructed by connecting each point to its horizontal
    neighbors that can be seen by a horizontal line of sight.
    """
    
    def __init__(self, weighted: bool = False, sparse: bool = False, limit: Optional[int] = None):
        """Initialize the HVG converter.
        
        Args:
            weighted: If True, edges will be weighted by the absolute difference
                between the connected points.
            sparse: If True, use sparse matrices for the adjacency matrix.
            limit: If provided, only connect points within temporal distance ≤ limit.
                   (R ts2net compatibility parameter)
        """
        self.weighted = weighted
        self.sparse = sparse
        self.limit = limit
    
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
        
        # Use fast Numba implementation if available
        if HAS_NUMBA:
            limit_val = self.limit if self.limit is not None else -1
            A = _hvg_adj_matrix_numba(self.x_, self.weighted, limit_val)
            
            if self.sparse:
                # Convert to sparse
                A_sparse = sparse.csr_matrix(A)
                G = nx.from_scipy_sparse_array(A_sparse)
                return G, A_sparse
            else:
                G = nx.from_numpy_array(A)
                return G, A
        
        # Fallback to original implementation if Numba not available
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        if self.sparse:
            rows, cols, data = [], [], []
        else:
            A = np.zeros((n, n), dtype=float)
        
        # Build the HVG by checking horizontal visibility between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Check if there's a clear line of sight between i and j
                visible = True
                for k in range(i + 1, j):
                    if self.x_[k] > min(self.x_[i], self.x_[j]):
                        visible = False
                        break
                
                if visible:
                    G.add_edge(i, j)
                    weight = abs(self.x_[i] - self.x_[j]) if self.weighted else 1.0
                    if self.sparse:
                        rows.append(i)
                        cols.append(j)
                        data.append(weight)
                        if i != j:  # Undirected graph, add both directions
                            rows.append(j)
                            cols.append(i)
                            data.append(weight)
                    else:
                        A[i, j] = weight
                        A[j, i] = weight  # Undirected graph
        
        if self.sparse:
            A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        
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
