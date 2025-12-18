"""
Recurrence Network implementation for time series analysis.

This module provides the RecurrenceNetwork class which converts time series into
complex networks using recurrence analysis.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Union, Tuple

# Third-party imports
try:
    from scipy.sparse import csr_matrix
    from scipy.spatial.distance import pdist, squareform
    import networkx as nx
except ImportError:
    csr_matrix = None
    nx = None

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


@njit(cache=True)
def _embed_numba(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Fast time-delay embedding using Numba.
    
    Args:
        x: 1D time series array
        m: Embedding dimension
        tau: Time delay
        
    Returns:
        Embedded matrix of shape (n_vectors, m)
    """
    n = len(x)
    k = (m - 1) * tau
    n_vectors = n - k
    
    # Pre-allocate output
    embedded = np.empty((n_vectors, m), dtype=x.dtype)
    
    for i in range(n_vectors):
        for j in range(m):
            embedded[i, j] = x[i + j * tau]
    
    return embedded


@njit(cache=True)
def _euclidean_dist_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of A and B.
    
    Args:
        A: Array of shape (n, d)
        B: Array of shape (m, d)
        
    Returns:
        Distance matrix of shape (n, m)
    """
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    
    D = np.empty((n, m), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = A[i, k] - B[j, k]
                dist_sq += diff * diff
            D[i, j] = np.sqrt(dist_sq)
    
    return D


@njit(cache=True)
def _manhattan_dist_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Manhattan distances between rows of A and B.
    
    Args:
        A: Array of shape (n, d)
        B: Array of shape (m, d)
        
    Returns:
        Distance matrix of shape (n, m)
    """
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    
    D = np.empty((n, m), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            dist = 0.0
            for k in range(d):
                dist += abs(A[i, k] - B[j, k])
            D[i, j] = dist
    
    return D


@njit(cache=True)
def _chebyshev_dist_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Chebyshev distances between rows of A and B.
    
    Args:
        A: Array of shape (n, d)
        B: Array of shape (m, d)
        
    Returns:
        Distance matrix of shape (n, m)
    """
    n = A.shape[0]
    m = B.shape[0]
    d = A.shape[1]
    
    D = np.empty((n, m), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            max_diff = 0.0
            for k in range(d):
                diff = abs(A[i, k] - B[j, k])
                if diff > max_diff:
                    max_diff = diff
            D[i, j] = max_diff
    
    return D


@dataclass
class RecurrenceNetwork:
    """Recurrence Network for time series analysis.

    Converts time series into complex networks using recurrence analysis.
    Supports both epsilon-neighborhood and k-nearest neighbors methods.

    Parameters:
    -----------
    m : int or None, default=None
        Embedding dimension for time-delay embedding.
        If None, automatically estimated using False Nearest Neighbors (FNN).
    tau : int, default=1
        Time delay for embedding
    rule : str, default="epsilon"
        Recurrence rule, either "epsilon" or "knn"
    epsilon : float, optional
        Threshold distance for epsilon-neighborhood rule
    k : int, default=10
        Number of nearest neighbors for knn rule
    target_density : float, optional
        Target density for automatic epsilon selection
    metric : str, default="euclidean"
        Distance metric ("euclidean", "manhattan", or "chebyshev")
    theiler : int, default=0
        Theiler window for excluding temporally close points
    include_self : bool, default=False
        Include self-loops in the recurrence matrix
    symmetrize : bool, default=True
        Make the adjacency matrix symmetric
    sparse : bool, default=False
        Use sparse matrices for large networks
    backend : str, default="rust"
        Backend to use ("rust" or "python")
    weighted : bool, default=False
        Use weighted edges
    """

    m: Optional[int] = None
    tau: int = 1
    rule: str = "epsilon"
    epsilon: Optional[float] = None
    k: int = 10
    target_density: Optional[float] = None
    metric: str = "euclidean"
    theiler: int = 0
    include_self: bool = False
    symmetrize: bool = True
    sparse: bool = False
    backend: str = "rust"
    weighted: bool = False

    def __post_init__(self):
        if self.rule not in ["epsilon", "knn"]:
            raise ValueError("rule must be 'epsilon' or 'knn'")
        
        # Normalize metric name (R compatibility: "maximum" → "chebyshev")
        _METRIC_ALIASES = {
            "euclidean": "euclidean",
            "manhattan": "manhattan",
            "chebyshev": "chebyshev",
            "maximum": "chebyshev",  # R ts2net uses "maximum" for Chebyshev
        }
        
        metric_lower = self.metric.lower()
        if metric_lower not in _METRIC_ALIASES:
            raise ValueError(
                f"Unknown metric: {self.metric}. "
                f"Must be one of: euclidean, manhattan, chebyshev, maximum"
            )
        
        self.metric = _METRIC_ALIASES[metric_lower]
        if (
            self.rule == "epsilon"
            and self.epsilon is None
            and self.target_density is None
        ):
            raise ValueError(
                "Must specify either epsilon or target_density for epsilon rule"
            )

    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between points."""
        _NUMBA_METRICS = {
            "euclidean": _euclidean_dist_numba,
            "manhattan": _manhattan_dist_numba,
            "chebyshev": _chebyshev_dist_numba,
        }
        
        _NUMPY_METRICS = {
            "euclidean": lambda A, B: np.linalg.norm(A[:, None] - B, axis=2),
            "manhattan": lambda A, B: np.sum(np.abs(A[:, None] - B), axis=2),
            "chebyshev": lambda A, B: np.max(np.abs(A[:, None] - B), axis=2),
        }
        
        metrics = _NUMBA_METRICS if HAS_NUMBA else _NUMPY_METRICS
        return metrics[self.metric](A, B)

    def _auto_eps(self, X: np.ndarray, q: float) -> float:
        """Automatically determine epsilon for target density q."""
        # Sample distances to estimate distribution
        n_samples = min(1000, X.shape[0])
        idx = np.random.choice(X.shape[0], size=n_samples, replace=False)
        D = self._dist(X[idx], X[idx])

        # Exclude diagonal and Theiler window
        mask = np.ones_like(D, dtype=bool)
        np.fill_diagonal(mask, False)
        if self.theiler > 0:
            mask &= np.abs(np.subtract.outer(idx, idx)) > self.theiler

        # Find epsilon for target density
        distances = D[mask].flatten()
        return float(np.quantile(distances, q))

    def _adj_epsilon(self, X: np.ndarray, eps: float) -> np.ndarray:
        """Create adjacency matrix using epsilon-neighborhood rule."""
        if self.backend == "rust":
            try:
                from ..core_rust import rn_adj_epsilon as _rn_adj_eps_rs

                return _rn_adj_eps_rs(
                    X,
                    eps,
                    self.theiler,
                    self.include_self,
                    self.symmetrize,
                    self.weighted,
                    self.metric,
                )
            except ImportError:
                self.backend = "python"

        # Python fallback
        D = self._dist(X, X)

        # Apply Theiler window
        if self.theiler > 0:
            mask = (
                np.abs(np.subtract.outer(np.arange(X.shape[0]), np.arange(X.shape[0])))
                <= self.theiler
            )
            D[mask] = np.inf

        # Create adjacency matrix
        A = (D <= eps).astype(float)

        if not self.include_self:
            np.fill_diagonal(A, 0)

        if self.symmetrize:
            A = np.maximum(A, A.T)

        if self.weighted:
            A = A * (1 - D / np.max(D))

        return A

    def _adj_knn(self, X: np.ndarray, k: int) -> np.ndarray:
        """Create adjacency matrix using k-nearest neighbors rule."""
        if self.backend == "rust":
            try:
                from ..core_rust import knn as _knn_rs

                return _knn_rs(
                    X,
                    k,
                    self.theiler,
                    self.include_self,
                    self.symmetrize,
                    self.weighted,
                    self.metric,
                )
            except ImportError:
                self.backend = "python"

        # Python fallback
        n = X.shape[0]
        D = self._dist(X, X)

        # Apply Theiler window
        if self.theiler > 0:
            mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= self.theiler
            D[mask] = np.inf

        # Find k-nearest neighbors (excluding self)
        if not self.include_self:
            np.fill_diagonal(D, np.inf)

        # Get indices of k nearest neighbors
        knn_indices = np.argpartition(D, k, axis=1)[:, :k]

        # Create adjacency matrix
        A = np.zeros((n, n))
        for i in range(n):
            A[i, knn_indices[i]] = 1

        if self.symmetrize:
            A = np.maximum(A, A.T)

        if self.weighted:
            # Use inverse distance as weight
            W = np.zeros_like(D)
            for i in range(n):
                W[i, knn_indices[i]] = 1 - D[i, knn_indices[i]] / np.max(
                    D[i, knn_indices[i]]
                )
            A = A * W

        return A

    def fit(self, x: np.ndarray) -> RecurrenceNetwork:
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
        x = np.asarray(x).squeeze()
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")

        # Store the input time series
        self.x_ = x

        # Auto-select embedding dimension if needed
        if self.m is None:
            try:
                from ..embedding import false_nearest_neighbors
                m_auto, _ = false_nearest_neighbors(x, tau=self.tau, max_dim=10)
                self.m = m_auto
                import logging
                logging.getLogger(__name__).info(f"Auto-selected embedding dimension: m={m_auto}")
            except ImportError:
                # Fallback if embedding module not available
                self.m = 3
                import logging
                logging.getLogger(__name__).warning("FNN not available, using default m=3")

        # Create time-delay embedding
        self.X_ = embed(x, self.m, self.tau)

        # Auto-select epsilon if needed
        if (
            self.rule == "epsilon"
            and self.epsilon is None
            and self.target_density is not None
        ):
            self.epsilon = self._auto_eps(self.X_, self.target_density)

        return self

    def transform(self) -> Tuple[nx.Graph, np.ndarray]:
        """Transform the input time series into a network.

        Returns:
        --------
        G : networkx.Graph or networkx.DiGraph
            The resulting graph.
        A : scipy.sparse.csr_matrix or numpy.ndarray
            The adjacency matrix of the graph.
        """
        if not hasattr(self, "x_"):
            raise RuntimeError("Must call fit() before transform()")

        if self.rule == "epsilon":
            A = self._adj_epsilon(self.X_, self.epsilon)
        else:  # knn
            A = self._adj_knn(self.X_, self.k)

        # Convert to networkx graph
        G = adj_to_graph(A, directed=not self.symmetrize)

        return G, A

    def fit_transform(self, x: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """Fit the model and transform the input time series.

        Parameters:
        -----------
        x : array-like, shape (n_samples,)
            Input time series.

        Returns:
        --------
        G : networkx.Graph or networkx.DiGraph
            The resulting graph.
        A : scipy.sparse.csr_matrix or numpy.ndarray
            The adjacency matrix of the graph.
        """
        return self.fit(x).transform()


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Create time-delay embedding of a time series.

    Parameters:
    -----------
    x : array-like, shape (n_samples,)
        Input time series.
    m : int
        Embedding dimension.
    tau : int
        Time delay.

    Returns:
    --------
    X : ndarray, shape (n_samples - (m-1)*tau, m)
        Time-delay embedding of the input time series.
    """
    n = len(x)
    k = (m - 1) * tau
    if k >= n:
        raise ValueError("Embedding dimension too large for time series length")
    
    # Use Numba-accelerated version if available
    if HAS_NUMBA:
        return _embed_numba(x, m, tau)
    
    # Fallback to list comprehension
    return np.asarray([x[i : i + k + 1 : tau] for i in range(n - k)])
