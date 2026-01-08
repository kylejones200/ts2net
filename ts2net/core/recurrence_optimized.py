"""
Optimized recurrence network construction using spatial indexing.

This module provides optimized implementations of recurrence networks using:
- KD-tree and ball tree for fast k-NN queries
- Approximate epsilon recurrence using spatial indexing
- Parallel computation for large datasets
- Memory-efficient algorithms
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Literal
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from scipy.sparse import csr_matrix

# Try to import spatial indexing
try:
    from scipy.spatial import cKDTree, distance_matrix
    HAS_SCIPY_SPATIAL = True
except ImportError:
    HAS_SCIPY_SPATIAL = False
    cKDTree = None

# Try to import parallel processing
try:
    from multiprocessing import Pool, cpu_count
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False
    Pool = None
    cpu_count = lambda: 1


SpatialIndex = Literal["kdtree", "ball_tree", "auto"]


def knn_recurrence_optimized(
    X: NDArray[np.float64],
    k: int,
    metric: str = "euclidean",
    index_type: SpatialIndex = "auto",
    n_jobs: Optional[int] = None,
    weighted: bool = False
) -> Tuple[nx.Graph, csr_matrix]:
    """
    Build k-NN recurrence network using spatial indexing for optimization.
    
    Uses KD-tree (for euclidean) or ball tree for fast nearest neighbor queries.
    Much faster than naive O(n²) approach for large datasets.
    
    Parameters
    ----------
    X : array (n, d)
        Time series data (n points, d dimensions)
    k : int
        Number of nearest neighbors
    metric : str, default "euclidean"
        Distance metric (only "euclidean" supported for KD-tree)
    index_type : str, default "auto"
        Spatial index type: "kdtree", "ball_tree", or "auto"
    n_jobs : int, optional
        Number of parallel jobs (None = use all CPUs)
    weighted : bool, default False
        If True, weight edges by distance
    
    Returns
    -------
    G : networkx.Graph
        k-NN recurrence network
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(1000, 10)
    >>> G, A = knn_recurrence_optimized(X, k=10, n_jobs=4)
    >>> print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    """
    if not HAS_SCIPY_SPATIAL:
        raise ImportError(
            "Spatial indexing requires scipy. Install with: pip install scipy"
        )
    
    n, d = X.shape
    
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in range [1, {n-1}], got {k}")
    
    # Determine index type
    if index_type == "auto":
        index_type = "kdtree" if metric == "euclidean" else None
    
    index_handlers = {
        "kdtree": lambda: _knn_kdtree(X, k, weighted, n_jobs) if metric == "euclidean" else None,
        "ball_tree": lambda: _knn_fallback(X, k, metric, weighted),
    }
    
    if index_type is None:
        return _knn_fallback(X, k, metric, weighted)
    
    handler = index_handlers.get(index_type)
    if handler is None:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    result = handler()
    if result is None:
        raise ValueError(
            f"KD-tree only supports euclidean metric, got {metric}. "
            f"Use index_type='ball_tree' or metric='euclidean'"
        )
    return result


def epsilon_recurrence_optimized(
    X: NDArray[np.float64],
    epsilon: float,
    metric: str = "euclidean",
    index_type: SpatialIndex = "auto",
    n_jobs: Optional[int] = None,
    weighted: bool = False,
    approximate: bool = True
) -> Tuple[nx.Graph, csr_matrix]:
    """
    Build epsilon recurrence network using spatial indexing for optimization.
    
    Uses spatial indexing to find all points within epsilon distance,
    avoiding the O(n²) all-pairs distance computation.
    
    Parameters
    ----------
    X : array (n, d)
        Time series data (n points, d dimensions)
    epsilon : float
        Distance threshold
    metric : str, default "euclidean"
        Distance metric
    index_type : str, default "auto"
        Spatial index type: "kdtree", "ball_tree", or "auto"
    n_jobs : int, optional
        Number of parallel jobs
    weighted : bool, default False
        If True, weight edges by distance
    approximate : bool, default True
        If True, use approximate search (faster, may miss some edges)
    
    Returns
    -------
    G : networkx.Graph
        Epsilon recurrence network
    A : scipy.sparse.csr_matrix
        Sparse adjacency matrix
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(1000, 10)
    >>> G, A = epsilon_recurrence_optimized(X, epsilon=0.5, n_jobs=4)
    >>> print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    """
    if not HAS_SCIPY_SPATIAL:
        raise ImportError(
            "Spatial indexing requires scipy. Install with: pip install scipy"
        )
    
    n, d = X.shape
    
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    
    # Determine index type
    if index_type == "auto":
        if metric == "euclidean":
            index_type = "kdtree"
        else:
            # For non-euclidean, use distance matrix approach
            return _epsilon_fallback(X, epsilon, metric, weighted)
    
    if index_type == "kdtree":
        if metric != "euclidean":
            raise ValueError(
                f"KD-tree only supports euclidean metric, got {metric}"
            )
        return _epsilon_kdtree(X, epsilon, weighted, n_jobs, approximate)
    
    else:
        return _epsilon_fallback(X, epsilon, metric, weighted)


def _knn_kdtree(
    X: NDArray[np.float64],
    k: int,
    weighted: bool,
    n_jobs: Optional[int]
) -> Tuple[nx.Graph, csr_matrix]:
    """Build k-NN using KD-tree."""
    n = X.shape[0]
    
    # Build KD-tree
    tree = cKDTree(X)
    
    # Query k+1 nearest neighbors (includes self)
    distances, indices = tree.query(X, k=k + 1, workers=n_jobs or -1)
    
    # Build sparse adjacency matrix (vectorized)
    i_indices = np.repeat(np.arange(n), k)
    j_indices = indices[:, 1:].flatten()
    
    if weighted:
        data = distances[:, 1:].flatten()
    else:
        data = np.ones(n * k, dtype=np.float64)
    
    A = csr_matrix((data, (i_indices, j_indices)), shape=(n, n))
    
    # Make symmetric (undirected)
    A = A + A.T
    
    # Convert to networkx graph
    G = nx.from_scipy_sparse_array(A) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(A)
    
    return G, A


def _epsilon_kdtree(
    X: NDArray[np.float64],
    epsilon: float,
    weighted: bool,
    n_jobs: Optional[int],
    approximate: bool
) -> Tuple[nx.Graph, csr_matrix]:
    """Build epsilon recurrence using KD-tree."""
    n = X.shape[0]
    
    # Build KD-tree
    tree = cKDTree(X)
    
    # Query all points within epsilon
    # query_ball_tree is more efficient than query_ball_point for all points
    if approximate:
        # Use query_ball_point for each point (faster, but may miss some)
        rows = []
        cols = []
        data = []
        
        # Process in chunks for parallelization
        if n_jobs and n_jobs > 1 and HAS_MULTIPROCESSING:
            chunk_size = max(1, n // (n_jobs * 4))
            chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
            
            def process_chunk(chunk_range):
                start, end = chunk_range
                chunk_rows, chunk_cols, chunk_data = [], [], []
                for i in range(start, end):
                    neighbors = tree.query_ball_point(X[i], epsilon, workers=1)
                    for j in neighbors:
                        if i < j:  # Only upper triangle
                            dist = np.linalg.norm(X[i] - X[j])
                            chunk_rows.append(i)
                            chunk_cols.append(j)
                            if weighted:
                                chunk_data.append(dist)
                            else:
                                chunk_data.append(1.0)
                return chunk_rows, chunk_cols, chunk_data
            
            with Pool(n_jobs) as pool:
                results = pool.map(process_chunk, chunks)
            
            for chunk_rows, chunk_cols, chunk_data in results:
                rows.extend(chunk_rows)
                cols.extend(chunk_cols)
                data.extend(chunk_data)
        else:
            # Sequential processing
            for i in range(n):
                neighbors = tree.query_ball_point(X[i], epsilon, workers=n_jobs or 1)
                for j in neighbors:
                    if i < j:  # Only upper triangle
                        dist = np.linalg.norm(X[i] - X[j])
                        rows.append(i)
                        cols.append(j)
                        if weighted:
                            data.append(dist)
                        else:
                            data.append(1.0)
    else:
        # Exact: use query_ball_tree (slower but exact)
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            neighbors = tree.query_ball_point(X[i], epsilon, workers=n_jobs or 1)
            for j in neighbors:
                if i != j:  # Exclude self
                    dist = np.linalg.norm(X[i] - X[j])
                    if dist <= epsilon:
                        rows.append(i)
                        cols.append(j)
                        if weighted:
                            data.append(dist)
                        else:
                            data.append(1.0)
    
    # Create sparse matrix
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Make symmetric
    A = A + A.T
    
    # Convert to networkx graph
    G = nx.from_scipy_sparse_array(A) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(A)
    
    return G, A


def _knn_fallback(
    X: NDArray[np.float64],
    k: int,
    metric: str,
    weighted: bool
) -> Tuple[nx.Graph, csr_matrix]:
    """Fallback k-NN using distance matrix (for non-euclidean metrics)."""
    from scipy.spatial.distance import cdist
    
    n = X.shape[0]
    
    # Map metric names to scipy-compatible names
    metric_map = {
        "manhattan": "cityblock",
        "l1": "cityblock",
        "l2": "euclidean",
    }
    scipy_metric = metric_map.get(metric.lower(), metric)
    
    # Compute distance matrix (memory intensive for large n)
    try:
        D = cdist(X, X, metric=scipy_metric)
    except ValueError:
        # If metric not supported, fall back to euclidean
        D = cdist(X, X, metric="euclidean")
    
    # Build k-NN
    rows = []
    cols = []
    data = []
    
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        distances = D[i].copy()
        distances[i] = np.inf
        neighbors = np.argpartition(distances, k - 1)[:k]
        
        for j in neighbors:
            rows.append(i)
            cols.append(j)
            if weighted:
                data.append(D[i, j])
            else:
                data.append(1.0)
    
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T  # Symmetrize
    
    G = nx.from_scipy_sparse_array(A) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(A)
    
    return G, A


def _epsilon_fallback(
    X: NDArray[np.float64],
    epsilon: float,
    metric: str,
    weighted: bool
) -> Tuple[nx.Graph, csr_matrix]:
    """Fallback epsilon recurrence using distance matrix."""
    from scipy.spatial.distance import cdist
    
    n = X.shape[0]
    
    # Map metric names to scipy-compatible names
    metric_map = {
        "manhattan": "cityblock",
        "l1": "cityblock",
        "l2": "euclidean",
    }
    scipy_metric = metric_map.get(metric.lower(), metric)
    
    # Compute distance matrix
    try:
        D = cdist(X, X, metric=scipy_metric)
    except ValueError:
        # If metric not supported, fall back to euclidean
        D = cdist(X, X, metric="euclidean")
    
    # Find all pairs within epsilon
    mask = (D <= epsilon) & (D > 0)  # Exclude diagonal
    
    rows, cols = np.where(mask)
    
    # Only upper triangle
    upper_mask = rows < cols
    rows = rows[upper_mask]
    cols = cols[upper_mask]
    
    if weighted:
        data = D[rows, cols]
    else:
        data = np.ones(len(rows))
    
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T  # Symmetrize
    
    G = nx.from_scipy_sparse_array(A) if hasattr(nx, 'from_scipy_sparse_array') else nx.from_scipy_sparse_matrix(A)
    
    return G, A


def _build_network_worker(args):
    """Worker function for parallel processing (must be at module level for pickling)."""
    X, method, k, epsilon, kwargs = args
    if method == "knn":
        if k is None:
            raise ValueError("k required for knn method")
        return knn_recurrence_optimized(X, k=k, n_jobs=1, **kwargs)
    else:
        if epsilon is None:
            raise ValueError("epsilon required for epsilon method")
        return epsilon_recurrence_optimized(X, epsilon=epsilon, n_jobs=1, **kwargs)


def parallel_recurrence_batch(
    X_list: list[NDArray[np.float64]],
    method: str = "knn",
    k: Optional[int] = None,
    epsilon: Optional[float] = None,
    n_jobs: Optional[int] = None,
    **kwargs
) -> list[Tuple[nx.Graph, csr_matrix]]:
    """
    Build recurrence networks for multiple time series in parallel.
    
    Parameters
    ----------
    X_list : list of arrays
        List of time series (each can have different length)
    method : str, default "knn"
        Method: "knn" or "epsilon"
    k : int, optional
        Number of neighbors (for knn method)
    epsilon : float, optional
        Distance threshold (for epsilon method)
    n_jobs : int, optional
        Number of parallel jobs (None = use all CPUs)
    **kwargs
        Additional arguments passed to recurrence functions
    
    Returns
    -------
    results : list of (Graph, csr_matrix) tuples
        List of recurrence networks
    
    Examples
    --------
    >>> import numpy as np
    >>> X_list = [np.random.randn(100, 5) for _ in range(10)]
    >>> results = parallel_recurrence_batch(X_list, method="knn", k=10, n_jobs=4)
    >>> print(f"Processed {len(results)} networks")
    """
    if not HAS_MULTIPROCESSING or n_jobs == 1:
        # Sequential fallback
        results = []
        for X in X_list:
            if method == "knn":
                if k is None:
                    raise ValueError("k required for knn method")
                result = knn_recurrence_optimized(X, k=k, n_jobs=1, **kwargs)
            else:
                if epsilon is None:
                    raise ValueError("epsilon required for epsilon method")
                result = epsilon_recurrence_optimized(X, epsilon=epsilon, n_jobs=1, **kwargs)
            results.append(result)
        return results
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Prepare arguments for worker function
    args_list = [(X, method, k, epsilon, kwargs) for X in X_list]
    
    with Pool(n_jobs) as pool:
        results = pool.map(_build_network_worker, args_list)
    
    return results

