"""Visibility graph implementations for time series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

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
def _hvg_adj_matrix_numba(x: np.ndarray) -> np.ndarray:
    """
    Build HVG adjacency matrix using stack-based algorithm with Numba.
    
    Args:
        x: 1D array of time series values
        
    Returns:
        A: Adjacency matrix of shape (n, n)
    """
    n = len(x)
    A = np.zeros((n, n), dtype=np.float64)
    
    stack = []
    for j in range(n):
        while len(stack) > 0 and x[stack[-1]] < x[j]:
            i = stack.pop()
            A[i, j] = 1.0
            A[j, i] = 1.0
        
        if len(stack) > 0:
            i = stack[-1]
            A[i, j] = 1.0
            A[j, i] = 1.0
        
        stack.append(j)
    
    return A


@njit(cache=True)
def _nvg_adj_matrix_numba(x: np.ndarray) -> np.ndarray:
    """
    Build NVG adjacency matrix using Numba acceleration.
    
    Args:
        x: 1D array of time series values
        
    Returns:
        A: Adjacency matrix of shape (n, n)
    """
    n = len(x)
    A = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            
            for k in range(i + 1, j):
                y_line = x[i] + (x[j] - x[i]) * (k - i) / (j - i)
                if x[k] > y_line:
                    visible = False
                    break
            
            if visible:
                A[i, j] = 1.0
                A[j, i] = 1.0
    
    return A


@dataclass
class VisibilityGraph:
    """Base class for visibility graphs."""
    
    def _visibility_condition(self, y1: float, y2: float, x1: int, x2: int, x: int, y: float) -> bool:
        """Check if two points are visible to each other.
        
        Args:
            y1: Y-coordinate of first point
            y2: Y-coordinate of second point
            x1: X-coordinate of first point
            x2: X-coordinate of second point
            x: X-coordinate of point to check visibility for
            y: Y-coordinate of point to check visibility for
            
        Returns:
            bool: True if points are visible to each other, False otherwise
        """
        return y <= y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    def fit(self, X: np.ndarray) -> 'VisibilityGraph':
        """Fit the visibility graph to the data.
        
        Args:
            X: Input time series data
            
        Returns:
            self: Returns the instance itself
        """
        self.X_ = X
        return self
    
    def transform(self) -> Union[nx.Graph, csr_matrix]:
        """Transform the time series into a visibility graph.
        
        Returns:
            Union[nx.Graph, csr_matrix]: The resulting visibility graph
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def fit_transform(self, X: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """Fit the model and transform the input data.
        
        Args:
            X: Input time series data
            
        Returns:
            Tuple[nx.Graph, np.ndarray]: The resulting visibility graph and adjacency matrix
        """
        G = self.fit(X).transform()
        A = nx.adjacency_matrix(G).toarray()
        return G, A


class HVG(VisibilityGraph):
    """Horizontal Visibility Graph (HVG) implementation."""
    
    def transform(self) -> Union[nx.Graph, csr_matrix]:
        """Transform the time series into a horizontal visibility graph.
        
        Returns:
            Union[nx.Graph, csr_matrix]: The resulting HVG
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Please fit the model first using .fit()")
            
        X = self.X_.flatten()
        n = len(X)
        
        # Use Numba-accelerated version if available
        if HAS_NUMBA:
            A = _hvg_adj_matrix_numba(X)
            G = nx.from_numpy_array(A)
            return G
        
        # Fallback to original implementation
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges based on horizontal visibility
        for i in range(n):
            for j in range(i + 1, n):
                # Check if all points between i and j are below the line connecting them
                visible = True
                for k in range(i + 1, j):
                    if X[k] >= min(X[i], X[j]) + (max(X[i], X[j]) - min(X[i], X[j])) * (k - i) / (j - i):
                        visible = False
                        break
                
                if visible:
                    G.add_edge(i, j)
        
        return G


class NVG(VisibilityGraph):
    """Natural Visibility Graph (NVG) implementation."""
    
    def transform(self) -> Union[nx.Graph, csr_matrix]:
        """Transform the time series into a natural visibility graph.
        
        Returns:
            Union[nx.Graph, csr_matrix]: The resulting NVG
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Please fit the model first using .fit()")
            
        X = self.X_.flatten()
        n = len(X)
        
        # Use Numba-accelerated version if available
        if HAS_NUMBA:
            A = _nvg_adj_matrix_numba(X)
            G = nx.from_numpy_array(A)
            return G
        
        # Fallback to original implementation
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # Add edges based on natural visibility
        for i in range(n):
            for j in range(i + 1, n):
                # Check if all points between i and j are below the line connecting them
                visible = True
                for k in range(i + 1, j):
                    if X[k] > X[i] + (X[j] - X[i]) * (k - i) / (j - i):
                        visible = False
                        break
                
                if visible:
                    G.add_edge(i, j)
        
        return G
