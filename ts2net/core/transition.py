"""Transition network implementation for time series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class TransitionNetwork:
    """A class representing a transition network.
    
    This class provides methods to create and analyze transition networks from time series data.
    """
    
    n_bins: int = 10
    order: int = 1
    # Parameter aliases for backward compatibility
    symbolizer: str = None  # symbolization method
    bins: int = None  # alias for n_bins
    delay: int = None  # time delay
    tie_rule: str = "stable"  # how to handle ties
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        # Handle parameter aliases
        if self.bins is not None:
            self.n_bins = self.bins
        if self.symbolizer is None:
            self.symbolizer = "equal_width"  # default symbolizer
            
        if self.n_bins < 2:
            raise ValueError("Number of bins must be at least 2")
        if self.order < 1:
            raise ValueError("Order must be at least 1")
    
    def _digitize(self, X: np.ndarray) -> np.ndarray:
        """Convert time series to symbolic representation.
        
        Args:
            X: Input time series data
            
        Returns:
            np.ndarray: Digitized time series
        """
        # Handle 1D input
        if X.ndim == 1:
            X = X.flatten()
            
        if self.symbolizer == "ordinal":
            # Use ordinal pattern encoding
            return self._ordinal_patterns(X)
        else:
            # Use equal-width binning (default)
            return np.digitize(
                X, 
                bins=np.linspace(X.min(), X.max(), self.n_bins + 1)[1:-1]
            )
    
    def _ordinal_patterns(self, X: np.ndarray) -> np.ndarray:
        """Convert time series to ordinal patterns.
        
        Args:
            X: Input time series data (1D)
            
        Returns:
            np.ndarray: Ordinal pattern indices
        """
        n = len(X)
        patterns = []
        
        for i in range(n - self.order + 1):
            # Get subsequence
            subseq = X[i:i + self.order]
            # Get ordinal pattern (rank order)
            pattern = tuple(np.argsort(subseq))
            patterns.append(pattern)
            
        # Convert patterns to unique indices
        unique_patterns = list(set(patterns))
        pattern_to_idx = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
        
        return np.array([pattern_to_idx[pattern] for pattern in patterns])
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences of specified order.
        
        Args:
            X: Input time series data
            
        Returns:
            np.ndarray: Array of sequences
        """
        sequences = []
        for i in range(len(X) - self.order):
            sequences.append(X[i:i + self.order + 1].flatten())
        return np.array(sequences)
    
    def fit(self, X: np.ndarray) -> 'TransitionNetwork':
        """Fit the transition network to the data.
        
        Args:
            X: Input time series data
            
        Returns:
            self: Returns the instance itself
        """
        self.X_ = X
        return self
    
    def transform(self) -> nx.DiGraph:
        """Transform the time series into a transition network.
        
        Returns:
            nx.DiGraph: The resulting transition network as a directed NetworkX graph
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Please fit the model first using .fit()")
            
        # Digitize the time series
        digitized = self._digitize(self.X_)
        
        # Create sequences
        sequences = self._create_sequences(digitized)
        
        # Create transition counts
        transition_counts = {}
        for seq in sequences:
            source = tuple(seq[:-1])
            target = tuple(seq[1:])
            
            if source not in transition_counts:
                transition_counts[source] = {}
            
            if target not in transition_counts[source]:
                transition_counts[source][target] = 0
                
            transition_counts[source][target] += 1
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges with weights
        for source, targets in transition_counts.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)
        
        return G
    
    def fit_transform(self, X: np.ndarray) -> Tuple[nx.DiGraph, np.ndarray]:
        """Fit the model and transform the input data.
        
        Args:
            X: Input time series data
            
        Returns:
            Tuple[nx.DiGraph, scipy.sparse.csr_matrix]: The resulting transition network 
                and sparse adjacency matrix (never dense)
        """
        from scipy import sparse as sp
        G = self.fit(X).transform()
        # Return sparse matrix, never dense
        A = nx.adjacency_matrix(G)
        # Convert to CSR if not already
        from scipy import sparse as sp
        if not isinstance(A, sp.csr_matrix):
            A = A.tocsr()
        return G, A
