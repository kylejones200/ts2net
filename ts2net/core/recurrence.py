"""Recurrence network implementation for time series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class RecurrenceNetwork:
    """A class representing a recurrence network.
    
    This class provides methods to create and analyze recurrence networks from time series data.
    """
    
    threshold: float = None
    embedding_dimension: int = None
    time_delay: int = 1
    metric: str = 'euclidean'
    # Parameter aliases for backward compatibility
    m: int = None  # alias for embedding_dimension
    tau: int = None  # alias for time_delay
    rule: str = None  # threshold rule
    target_density: float = None  # for density-based thresholding
    k: int = None  # for k-NN rule
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        # Handle parameter aliases
        if self.m is not None:
            self.embedding_dimension = self.m
        if self.tau is not None:
            self.time_delay = self.tau
            
        # Set defaults if not provided
        if self.embedding_dimension is None:
            self.embedding_dimension = 1
        if self.threshold is None and self.target_density is not None:
            # Will be computed during fit based on target_density
            pass
        elif self.threshold is None:
            self.threshold = 0.1  # default threshold
            
        if self.embedding_dimension < 1:
            raise ValueError("Embedding dimension must be at least 1")
        if self.time_delay < 1:
            raise ValueError("Time delay must be at least 1")
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
    
    def fit(self, X: np.ndarray) -> 'RecurrenceNetwork':
        """Fit the recurrence network to the data.
        
        Args:
            X: Input time series data
            
        Returns:
            self: Returns the instance itself
        """
        self.X_ = X
        return self
    
    def transform(self) -> Union[nx.Graph, csr_matrix]:
        """Transform the time series into a recurrence network.
        
        Returns:
            Union[nx.Graph, csr_matrix]: The resulting recurrence network as a NetworkX graph
                                         or sparse matrix
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Please fit the model first using .fit()")
            
        # Build edges directly (avoid dense distance matrix)
        n = len(self.X_)
        
        # Safety guardrail: refuse exact all-pairs for large n
        if n > 50_000 and self.rule == 'epsilon':
            raise ValueError(
                f"Refusing exact all-pairs recurrence for n={n} nodes. "
                f"This would require ~{n**2 * 8 / 1e9:.1f} GB for distance matrix. "
                f"Use rule='knn' with small k (e.g., k=10-30) instead, or resample the series."
            )
        
        # Build edges directly without full distance matrix
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.X_[i] - self.X_[j])
                if dist <= self.threshold:
                    edges.append((i, j))
        
        # Convert to networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        return G
    
    def fit_transform(self, X: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """Fit the model and transform the input data.
        
        Args:
            X: Input time series data
            
        Returns:
            Tuple[nx.Graph, scipy.sparse.csr_matrix]: The resulting recurrence network 
                and sparse adjacency matrix (never dense)
        """
        from scipy import sparse as sp
        G = self.fit(X).transform()
        # Return sparse matrix, never dense
        A = nx.adjacency_matrix(G, format='csr')
        return G, A
