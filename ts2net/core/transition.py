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
    
    For ordinal partition networks, set partition_mode=True and symbolizer="ordinal" to enable
    entropy rate computation and pattern motif counting.
    """
    
    n_bins: int = 10
    order: int = 1
    # Parameter aliases for backward compatibility
    symbolizer: str = None  # symbolization method
    bins: int = None  # alias for n_bins
    delay: int = None  # time delay
    tie_rule: str = "stable"  # how to handle ties
    partition_mode: bool = False  # Enable partition-based analysis (entropy, motifs)
    
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
        self._G = None  # Cache for graph
        self._pattern_distribution = None  # Cache for pattern distribution
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
        
        self._G = G  # Cache graph for partition mode methods
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
        self._G = G  # Cache graph for partition mode methods
        # Return sparse matrix, never dense
        A = nx.adjacency_matrix(G)
        # Convert to CSR if not already
        if not isinstance(A, sp.csr_matrix):
            A = A.tocsr()
        return G, A
    
    def entropy_rate(self) -> float:
        """Compute entropy rate of the transition network.
        
        The entropy rate measures the complexity of pattern transitions.
        Higher values indicate more complex/random patterns.
        
        Returns:
            float: Entropy rate in bits
        """
        if not hasattr(self, '_G') or self._G is None:
            raise ValueError("Must call fit_transform() or transform() first")
        
        if not self.partition_mode or self.symbolizer != "ordinal":
            raise ValueError(
                "entropy_rate() requires partition_mode=True and symbolizer='ordinal'"
            )
        
        # Compute transition probabilities
        total_out_weight = {}
        for node in self._G.nodes():
            total_out_weight[node] = sum(
                data.get('weight', 1) for _, _, data in self._G.out_edges(node, data=True)
            )
        
        # Compute entropy rate: H = -sum(p_i * sum(p_ij * log2(p_ij)))
        entropy = 0.0
        for node in self._G.nodes():
            out_weight = total_out_weight.get(node, 0)
            if out_weight == 0:
                continue
            
            # Stationary probability of being at this node (proportional to in-degree weight)
            in_weight = sum(
                data.get('weight', 1) for _, _, data in self._G.in_edges(node, data=True)
            )
            total_weight = sum(
                data.get('weight', 1) for _, _, data in self._G.edges(data=True)
            )
            if total_weight == 0:
                continue
            
            pi = in_weight / total_weight if total_weight > 0 else 0
            
            # Conditional entropy from this node
            node_entropy = 0.0
            for _, target, data in self._G.out_edges(node, data=True):
                weight = data.get('weight', 1)
                p_ij = weight / out_weight if out_weight > 0 else 0
                if p_ij > 0:
                    node_entropy -= p_ij * np.log2(p_ij)
            
            entropy += pi * node_entropy
        
        return float(entropy)
    
    def pattern_distribution(self) -> dict:
        """Get frequency distribution of ordinal patterns.
        
        Returns:
            dict: Dictionary mapping pattern indices to frequencies
        """
        if not hasattr(self, 'X_'):
            raise ValueError("Must call fit() first")
        
        if not self.partition_mode or self.symbolizer != "ordinal":
            raise ValueError(
                "pattern_distribution() requires partition_mode=True and symbolizer='ordinal'"
            )
        
        if self._pattern_distribution is None:
            # Get ordinal patterns
            patterns = self._ordinal_patterns(self.X_)
            
            # Count frequencies
            unique, counts = np.unique(patterns, return_counts=True)
            total = len(patterns)
            self._pattern_distribution = {
                int(idx): float(count / total) for idx, count in zip(unique, counts)
            }
        
        return self._pattern_distribution.copy()
    
    def pattern_motifs(self, motif_type: str = '3node') -> dict:
        """Count motifs in the ordinal partition network.
        
        Args:
            motif_type: Type of motifs to count ('3node' or '4node')
            
        Returns:
            dict: Dictionary of motif counts
        """
        if not hasattr(self, '_G') or self._G is None:
            raise ValueError("Must call fit_transform() or transform() first")
        
        if not self.partition_mode or self.symbolizer != "ordinal":
            raise ValueError(
                "pattern_motifs() requires partition_mode=True and symbolizer='ordinal'"
            )
        
        # Import motif counting functions
        from ts2net.networks.motifs import directed_3node_motifs, undirected_4node_motifs
        
        if motif_type == '3node':
            # For directed graphs, use directed 3-node motifs
            return directed_3node_motifs(self._G)
        elif motif_type == '4node':
            # Convert to undirected for 4-node motifs
            G_und = self._G.to_undirected()
            return undirected_4node_motifs(G_und)
        else:
            raise ValueError(f"motif_type must be '3node' or '4node', got {motif_type}")
    
    def stats(self) -> dict:
        """Get summary statistics including partition mode metrics.
        
        Returns:
            dict: Dictionary of statistics including entropy_rate, pattern_distribution,
                  and motif_counts if partition_mode is enabled
        """
        if not hasattr(self, '_G') or self._G is None:
            raise ValueError("Must call fit_transform() or transform() first")
        
        stats = {
            'n_nodes': self._G.number_of_nodes(),
            'n_edges': self._G.number_of_edges(),
            'density': nx.density(self._G),
        }
        
        # Add partition mode statistics
        if self.partition_mode and self.symbolizer == "ordinal":
            try:
                stats['entropy_rate'] = self.entropy_rate()
                stats['pattern_distribution'] = self.pattern_distribution()
                
                # Add motif counts
                try:
                    stats['motif_counts_3node'] = self.pattern_motifs('3node')
                except Exception:
                    stats['motif_counts_3node'] = {}
                
                try:
                    stats['motif_counts_4node'] = self.pattern_motifs('4node')
                except Exception:
                    stats['motif_counts_4node'] = {}
            except Exception as e:
                # If partition mode metrics fail, still return basic stats
                stats['partition_mode_error'] = str(e)
        
        return stats
