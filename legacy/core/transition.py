"""
Transition Network implementation for time series analysis.

This module provides the TransitionNetwork class which converts a time series into
a directed graph where nodes represent states and edges represent transitions
between states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

# Third-party imports
try:
    import networkx as nx
    from scipy import sparse
except ImportError:
    nx = None
    sparse = None

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

# Local imports
from .utils.graph import adj_to_graph


class SKMixin:
    """Mixin for scikit-learn compatibility."""
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not k.endswith('_')}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


@njit(cache=True)
def _ordinal_patterns_numba(x: np.ndarray, order: int, delay: int) -> np.ndarray:
    """
    Fast ordinal pattern computation using Numba.
    
    Converts time series windows into ordinal patterns by ranking values.
    Uses stable sorting (preserves order of equal elements).
    
    Args:
        x: 1D time series array
        order: Number of points in each pattern
        delay: Time delay between points
        
    Returns:
        Array of ordinal pattern indices
    """
    n = len(x)
    k = (order - 1) * delay
    n_patterns = n - k
    
    patterns = np.empty(n_patterns, dtype=np.int64)
    
    # Pre-compute factorials
    factorials = np.empty(order, dtype=np.int64)
    factorials[0] = 1
    for i in range(1, order):
        factorials[i] = factorials[i - 1] * i
    
    # Extract patterns
    for i in range(n_patterns):
        # Get the window
        window = np.empty(order, dtype=np.float64)
        for j in range(order):
            window[j] = x[i + j * delay]
        
        # Compute ranks using stable sort (argsort)
        # We need to implement stable argsort manually for Numba
        ranks = np.empty(order, dtype=np.int64)
        
        # Simple insertion sort for stable ranking (good for small order)
        indices = np.arange(order)
        for j in range(1, order):
            key_val = window[j]
            key_idx = indices[j]
            k_pos = j - 1
            
            # Stable: only move if strictly less than (not <=)
            while k_pos >= 0 and window[indices[k_pos]] > key_val:
                indices[k_pos + 1] = indices[k_pos]
                k_pos -= 1
            indices[k_pos + 1] = key_idx
        
        # Convert sorted indices to ranks
        for j in range(order):
            ranks[indices[j]] = j
        
        # Convert permutation to unique integer using factorial number system
        pattern_code = 0
        used = np.zeros(order, dtype=np.int64)
        
        for j in range(order):
            r = ranks[j]
            # Count how many smaller unused ranks exist
            smaller_count = 0
            for m in range(r):
                if used[m] == 0:
                    smaller_count += 1
            
            pattern_code += smaller_count * factorials[order - j - 1]
            used[r] = 1
        
        patterns[i] = pattern_code
    
    return patterns


@njit(cache=True)
def _build_transition_matrix_numba(symbols: np.ndarray, n_symbols: int, 
                                   normalize: bool) -> np.ndarray:
    """
    Build transition matrix from symbol sequence using Numba.
    
    Args:
        symbols: Array of integer symbols
        n_symbols: Number of unique symbols
        normalize: Whether to normalize to probabilities
        
    Returns:
        Transition matrix of shape (n_symbols, n_symbols)
    """
    A = np.zeros((n_symbols, n_symbols), dtype=np.float64)
    
    # Count transitions
    for i in range(len(symbols) - 1):
        from_state = symbols[i]
        to_state = symbols[i + 1]
        A[from_state, to_state] += 1.0
    
    # Normalize if requested
    if normalize:
        for i in range(n_symbols):
            row_sum = 0.0
            for j in range(n_symbols):
                row_sum += A[i, j]
            
            if row_sum > 0:
                for j in range(n_symbols):
                    A[i, j] /= row_sum
    
    return A


@dataclass
class TransitionNetwork(SKMixin):
    """Transition Network for time series analysis.

    Converts a time series into a directed graph where nodes represent states
    and edges represent transitions between states. Supports different methods
    for symbolizing the time series before constructing the transition network.

    Parameters:
    -----------
    symbolizer : str, default="ordinal"
        Method for converting the time series into symbols. Options are:
        - "ordinal": Use ordinal patterns (recommended for most cases)
        - "equal_width": Equal-width binning
        - "equal_freq": Equal-frequency binning
        - "kmeans": K-means clustering
    order : int, default=3
        For ordinal patterns, the number of points to include in each pattern.
    delay : int, default=1
        For ordinal patterns, the time delay between points in a pattern.
    tie_rule : str, default="stable"
        For ordinal patterns, how to handle ties in the data. Options are:
        - "stable": Preserve the order of tied values (recommended)
        - "random": Randomly break ties
    bins : int, default=5
        Number of bins/symbols to use for binning-based symbolizers.
    normalize : bool, default=True
        If True, normalize the transition matrix to represent probabilities.
    sparse : bool, default=False
        If True, use sparse matrices for the transition matrix.
    """

    symbolizer: Literal["ordinal", "equal_width", "equal_freq", "kmeans"]
    order: int
    delay: int
    tie_rule: Literal["stable", "random"]
    bins: int
    normalize: bool
    sparse: bool

    def __post_init__(self) -> None:
        """Validate parameters after initialization.

        Raises:
            ValueError: If symbolizer or tie_rule has an invalid value.
        """
        valid_symbolizers = ["ordinal", "equal_width", "equal_freq", "quantiles", "kmeans"]
        if self.symbolizer not in valid_symbolizers:
            raise ValueError(
                f"symbolizer must be one of: {valid_symbolizers}"
            )
        if self.tie_rule not in ["stable", "random"]:
            raise ValueError("tie_rule must be 'stable' or 'random'")

    def fit(self, x: np.ndarray) -> "TransitionNetwork":
        """Fit the model to the input time series.

        Parameters:
        -----------
        x : array-like, shape (n_samples,)
            Input time series.

        Returns:
        --------
        self : object
            Returns self.
        """
        self.x_ = np.asarray(x).squeeze()
        if self.x_.ndim != 1:
            raise ValueError("Input must be a 1D array")

        # Store the symbolized time series
        self.symbols_ = self._symbols()

        return self

    def _symbols(self) -> NDArray[np.int64]:
        """Convert the time series into symbolic representation."""
        
        def _ordinal():
            return _ordinal_patterns(self.x_, self.order, self.delay, self.tie_rule)
        
        def _binned_base():
            return (self.x_ - np.min(self.x_)) / (np.max(self.x_) - np.min(self.x_) + 1e-10)
        
        def _equal_width():
            x_norm = _binned_base()
            return np.digitize(x_norm, np.linspace(0, 1, self.bins + 1)[1:-1])
        
        def _equal_freq():
            x_norm = _binned_base()
            return np.digitize(x_norm, np.percentile(x_norm, np.linspace(0, 100, self.bins + 1)[1:-1]))
        
        def _kmeans():
            from sklearn.cluster import KMeans
            x_norm = _binned_base()
            return KMeans(n_clusters=self.bins, n_init=10).fit_predict(x_norm.reshape(-1, 1))
        
        _SYMBOLIZERS = {
            "ordinal": _ordinal,
            "equal_width": _equal_width,
            "equal_freq": _equal_freq,
            "quantiles": _equal_freq,  # R compatibility: quantiles = equal_freq
            "kmeans": _kmeans,
        }
        
        symbolizer_fn = _SYMBOLIZERS.get(self.symbolizer)
        if symbolizer_fn is None:
            raise ValueError(f"Unknown symbolizer: {self.symbolizer}")
        
        return symbolizer_fn()

    def transform(
        self,
    ) -> Tuple[nx.DiGraph, Union[sparse.csr_matrix, NDArray[np.float64]]]:
        """Transform the symbolized time series into a transition network.

        Returns:
            A tuple containing:
                - G: The transition graph as a networkx.DiGraph
                - A: The transition matrix as either a sparse or dense array

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        """Transform the symbolized time series into a transition network.
        
        Returns:
        --------
        G : networkx.DiGraph
            The transition graph.
        A : scipy.sparse.csr_matrix or numpy.ndarray
            The transition matrix.
        """
        if not hasattr(self, "symbols_"):
            raise RuntimeError("Must call fit() before transform()")

        # Get unique symbols and create mapping to indices
        unique_symbols = np.unique(self.symbols_)
        n_symbols = len(unique_symbols)
        symbol_to_idx = {s: i for i, s in enumerate(unique_symbols)}
        
        # Map symbols to indices
        indexed_symbols = np.array([symbol_to_idx[s] for s in self.symbols_], dtype=np.int64)

        # Use Numba-accelerated transition matrix building if available and not sparse
        if HAS_NUMBA and not self.sparse:
            A = _build_transition_matrix_numba(indexed_symbols, n_symbols, self.normalize)
        else:
            # Create transition matrix
            if self.sparse:
                A = sparse.lil_matrix((n_symbols, n_symbols), dtype=float)
            else:
                A = np.zeros((n_symbols, n_symbols), dtype=float)

            # Count transitions
            for i in range(len(indexed_symbols) - 1):
                from_state = indexed_symbols[i]
                to_state = indexed_symbols[i + 1]
                A[from_state, to_state] += 1

            # Normalize to get probabilities if requested
            if self.normalize:
                row_sums = A.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                A = A / row_sums

        # Convert to final sparse format if needed
        if self.sparse and not sparse.isspmatrix_csr(A):
            A = A.tocsr()

        # Create graph
        G = adj_to_graph(A, directed=True)

        # Add node attributes for the symbol values
        for i, s in enumerate(unique_symbols):
            G.nodes[i]["symbol"] = s

        return G, A

    def fit_transform(
        self, x: np.ndarray
    ) -> Tuple[nx.DiGraph, Union[sparse.csr_matrix, np.ndarray]]:
        """Fit the model and transform the input time series.

        Parameters:
        -----------
        x : array-like, shape (n_samples,)
            Input time series.

        Returns:
        --------
        G : networkx.DiGraph
            The transition graph.
        A : scipy.sparse.csr_matrix or numpy.ndarray
            The transition matrix.
        """
        return self.fit(x).transform()


def _ordinal_patterns(
    x: NDArray[np.float64], order: int, delay: int = 1, tie_rule: str = "stable"
) -> NDArray[np.int64]:
    """Convert time series to ordinal patterns.

    Parameters:
    -----------
    x : array-like, shape (n_samples,)
        Input time series.
    order : int
        Number of points to include in each pattern.
    delay : int, default=1
        Time delay between points in a pattern.
    tie_rule : str, default="stable"
        How to handle ties in the data. Options are:
        - "stable": Preserve the order of tied values (recommended)
        - "random": Randomly break ties

    Returns:
    --------
    patterns : ndarray, shape (n_samples - (order-1)*delay,)
        Array of ordinal pattern indices.
    """
    # Use Numba-accelerated version for stable tie rule
    if HAS_NUMBA and tie_rule == "stable":
        return _ordinal_patterns_numba(x, order, delay)
    
    # Fallback to original implementation
    n = len(x)
    k = (order - 1) * delay

    # Create the embedding matrix
    indices = np.arange(n - k)
    for i in range(1, order):
        indices = np.column_stack((indices, np.arange(i * delay, n - k + i * delay)))

    # Get the embedded vectors
    embedded = x[indices]

    # Sort each vector to get the permutation pattern
    if tie_rule == "random":
        # Add small random noise to break ties randomly
        rng = np.random.RandomState(42)
        embedded = embedded + 1e-10 * rng.randn(*embedded.shape)

    # Get the permutation pattern for each vector
    patterns = np.argsort(embedded, axis=1)

    # Convert permutation pattern to a unique integer
    factor = np.array([np.math.factorial(i) for i in range(order)])
    pattern_indices = np.sum(patterns * factor, axis=1)

    return pattern_indices
