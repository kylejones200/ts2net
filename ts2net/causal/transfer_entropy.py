"""
Transfer entropy for causal inference.

Transfer entropy is an information-theoretic measure that quantifies the amount
of information transferred from one time series to another, providing a measure
of causal influence.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Union, Literal
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from scipy.sparse import csr_matrix

from ..core.graph import Graph
from ._parallel import pairwise_parallel


def _discretize_series(x: NDArray[np.float64], bins: int = 10) -> NDArray[np.int32]:
    """Discretize continuous time series into bins."""
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if x_min == x_max:
        return np.zeros(len(x), dtype=np.int32)
    
    edges = np.linspace(x_min, x_max, bins + 1)
    edges[-1] += 1e-10  # Ensure max value is included
    return np.digitize(x, edges) - 1


def _compute_entropy(probs: NDArray[np.float64]) -> float:
    """Compute Shannon entropy from probability distribution."""
    probs = probs[probs > 0]  # Remove zeros
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _compute_joint_entropy(x: NDArray, y: NDArray, bins: int = 10) -> float:
    """Compute joint entropy H(X, Y)."""
    x_disc = _discretize_series(x, bins)
    y_disc = _discretize_series(y, bins)
    
    joint_counts = np.zeros((bins, bins), dtype=np.int32)
    for i in range(len(x_disc)):
        if 0 <= x_disc[i] < bins and 0 <= y_disc[i] < bins:
            joint_counts[x_disc[i], y_disc[i]] += 1
    
    total = np.sum(joint_counts)
    if total == 0:
        return 0.0
    
    joint_probs = joint_counts / total
    return _compute_entropy(joint_probs.flatten())


def _compute_conditional_entropy(
    x: NDArray, 
    y: NDArray, 
    z: Optional[NDArray] = None,
    bins: int = 10
) -> float:
    """Compute conditional entropy H(X | Y) or H(X | Y, Z)."""
    if z is None:
        x_disc = _discretize_series(x, bins)
        y_disc = _discretize_series(y, bins)
        
        joint_counts = np.zeros((bins, bins), dtype=np.int32)
        for i in range(len(x_disc)):
            if 0 <= x_disc[i] < bins and 0 <= y_disc[i] < bins:
                joint_counts[x_disc[i], y_disc[i]] += 1
        
        joint_probs = joint_counts / (np.sum(joint_counts) + 1e-10)
        joint_entropy = _compute_entropy(joint_probs.flatten())
        
        y_counts = np.bincount(y_disc[y_disc >= 0], minlength=bins)
        y_probs = y_counts / (np.sum(y_counts) + 1e-10)
        y_entropy = _compute_entropy(y_probs)
        
        return joint_entropy - y_entropy
    
    x_disc = _discretize_series(x, bins)
    y_disc = _discretize_series(y, bins)
    z_disc = _discretize_series(z, bins)
    
    joint_counts = np.zeros((bins, bins, bins), dtype=np.int32)
    for i in range(len(x_disc)):
        if (0 <= x_disc[i] < bins and 0 <= y_disc[i] < bins and 
            0 <= z_disc[i] < bins):
            joint_counts[x_disc[i], y_disc[i], z_disc[i]] += 1
    
    joint_probs = joint_counts / (np.sum(joint_counts) + 1e-10)
    joint_entropy = _compute_entropy(joint_probs.flatten())
    
    marginal_counts = np.zeros((bins, bins), dtype=np.int32)
    for i in range(len(y_disc)):
        if 0 <= y_disc[i] < bins and 0 <= z_disc[i] < bins:
            marginal_counts[y_disc[i], z_disc[i]] += 1
    
    marginal_probs = marginal_counts / (np.sum(marginal_counts) + 1e-10)
    marginal_entropy = _compute_entropy(marginal_probs.flatten())
    
    return joint_entropy - marginal_entropy


def transfer_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lag: int = 1,
    bins: int = 10,
    method: Literal["discrete", "knn"] = "discrete"
) -> float:
    """
    Compute transfer entropy from X to Y.
    
    Transfer entropy measures the amount of information transferred from X to Y,
    quantifying causal influence. TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}).
    
    Parameters
    ----------
    x : array (n,)
        Source time series
    y : array (n,)
        Target time series (must have same length as x)
    lag : int, default 1
        Time lag for past values
    bins : int, default 10
        Number of bins for discretization (only used if method="discrete")
    method : str, default "discrete"
        Method for computing entropy: "discrete" (binning) or "knn" (k-nearest neighbors)
    
    Returns
    -------
    te : float
        Transfer entropy from X to Y (non-negative, bits)
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> y = x[:-1] + 0.1 * np.random.randn(999)  # y depends on x
    >>> y = np.concatenate([[0], y])  # Pad to same length
    >>> te = transfer_entropy(x, y, lag=1)
    >>> print(f"Transfer entropy X→Y: {te:.4f} bits")
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")
    
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    
    if len(x) < lag + 1:
        return 0.0
    
    if method == "discrete":
        y_t = y[lag:]
        y_past = y[:-lag] if lag > 0 else y
        x_past = x[:-lag] if lag > 0 else x
        
        if len(y_t) != len(y_past) or len(y_t) != len(x_past):
            min_len = min(len(y_t), len(y_past), len(x_past))
            y_t = y_t[:min_len]
            y_past = y_past[:min_len]
            x_past = x_past[:min_len]
        
        h_y_given_y_past = _compute_conditional_entropy(y_t, y_past, bins=bins)
        h_y_given_y_past_x_past = _compute_conditional_entropy(
            y_t, y_past, z=x_past, bins=bins
        )
        
        te = h_y_given_y_past - h_y_given_y_past_x_past
        return max(0.0, te)  # Transfer entropy is non-negative
    
    elif method == "knn":
        try:
            from scipy.spatial.distance import cdist
            from scipy.stats import entropy as scipy_entropy
        except ImportError:
            raise ImportError(
                "knn method requires scipy. Install with: pip install scipy"
            )
        
        y_t = y[lag:].reshape(-1, 1)
        y_past = y[:-lag].reshape(-1, 1) if lag > 0 else y.reshape(-1, 1)
        x_past = x[:-lag].reshape(-1, 1) if lag > 0 else x.reshape(-1, 1)
        
        min_len = min(len(y_t), len(y_past), len(x_past))
        y_t = y_t[:min_len]
        y_past = y_past[:min_len]
        x_past = x_past[:min_len]
        
        k = max(3, int(np.sqrt(min_len)))
        
        joint = np.hstack([y_t, y_past, x_past])
        marginal = np.hstack([y_t, y_past])
        
        dists_joint = cdist(joint, joint)
        dists_marginal = cdist(marginal, marginal)
        
        np.fill_diagonal(dists_joint, np.inf)
        np.fill_diagonal(dists_marginal, np.inf)
        
        eps_joint = np.partition(dists_joint, k, axis=1)[:, k]
        eps_marginal = np.partition(dists_marginal, k, axis=1)[:, k]
        
        n_joint = np.sum(dists_joint <= eps_joint[:, None], axis=1)
        n_marginal = np.sum(dists_marginal <= eps_marginal[:, None], axis=1)
        
        te = np.mean(np.log2(n_marginal / n_joint))
        return max(0.0, te)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'discrete' or 'knn'")


def conditional_transfer_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    lag: int = 1,
    bins: int = 10
) -> float:
    """
    Compute conditional transfer entropy from X to Y given Z.
    
    Conditional transfer entropy accounts for confounding variables Z,
    measuring the direct causal influence from X to Y.
    
    CTE(X→Y|Z) = H(Y_t | Y_{t-1}, Z_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}, Z_{t-1})
    
    Parameters
    ----------
    x : array (n,)
        Source time series
    y : array (n,)
        Target time series
    z : array (n,)
        Conditioning time series (confounding variable)
    lag : int, default 1
        Time lag for past values
    bins : int, default 10
        Number of bins for discretization
    
    Returns
    -------
    cte : float
        Conditional transfer entropy from X to Y given Z (non-negative, bits)
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1000)
    >>> z = np.random.randn(1000)
    >>> y = 0.5 * x[:-1] + 0.3 * z[:-1] + 0.1 * np.random.randn(999)
    >>> y = np.concatenate([[0], y])
    >>> cte = conditional_transfer_entropy(x, y, z, lag=1)
    >>> print(f"Conditional transfer entropy X→Y|Z: {cte:.4f} bits")
    """
    if not (len(x) == len(y) == len(z)):
        raise ValueError("All series must have same length")
    
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    
    if len(x) < lag + 1:
        return 0.0
    
    y_t = y[lag:]
    y_past = y[:-lag] if lag > 0 else y
    x_past = x[:-lag] if lag > 0 else x
    z_past = z[:-lag] if lag > 0 else z
    
    min_len = min(len(y_t), len(y_past), len(x_past), len(z_past))
    y_t = y_t[:min_len]
    y_past = y_past[:min_len]
    x_past = x_past[:min_len]
    z_past = z_past[:min_len]
    
    h_y_given_yz = _compute_conditional_entropy(y_t, y_past, z=z_past, bins=bins)
    
    y_past_disc = _discretize_series(y_past, bins)
    x_past_disc = _discretize_series(x_past, bins)
    z_past_disc = _discretize_series(z_past, bins)
    y_t_disc = _discretize_series(y_t, bins)
    
    joint_counts = np.zeros((bins, bins, bins, bins), dtype=np.int32)
    for i in range(len(y_t_disc)):
        if (0 <= y_t_disc[i] < bins and 0 <= y_past_disc[i] < bins and
            0 <= x_past_disc[i] < bins and 0 <= z_past_disc[i] < bins):
            joint_counts[y_t_disc[i], y_past_disc[i], x_past_disc[i], z_past_disc[i]] += 1
    
    joint_probs = joint_counts / (np.sum(joint_counts) + 1e-10)
    joint_entropy = _compute_entropy(joint_probs.flatten())
    
    marginal_counts = np.zeros((bins, bins, bins), dtype=np.int32)
    for i in range(len(y_past_disc)):
        if (0 <= y_past_disc[i] < bins and 0 <= x_past_disc[i] < bins and
            0 <= z_past_disc[i] < bins):
            marginal_counts[y_past_disc[i], x_past_disc[i], z_past_disc[i]] += 1
    
    marginal_probs = marginal_counts / (np.sum(marginal_counts) + 1e-10)
    marginal_entropy = _compute_entropy(marginal_probs.flatten())
    
    h_y_given_yxz = joint_entropy - marginal_entropy
    
    cte = h_y_given_yz - h_y_given_yxz
    return max(0.0, cte)


def transfer_entropy_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    lag: int = 1,
    bins: int = 10,
    threshold: Optional[float] = None,
    method: Literal["discrete", "knn"] = "discrete",
    series_names: Optional[List[str]] = None,
    n_jobs: int = 1,
) -> Tuple[nx.DiGraph, NDArray, Dict[str, float]]:
    """
    Construct a directed network based on transfer entropy between time series.
    
    Each edge (i, j) represents causal influence from series i to series j,
    weighted by the transfer entropy value.
    
    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Multiple time series to analyze
    lag : int, default 1
        Time lag for transfer entropy computation
    bins : int, default 10
        Number of bins for discretization (only used if method="discrete")
    threshold : float, optional
        Minimum transfer entropy threshold for edges (if None, include all edges)
    method : str, default "discrete"
        Method for computing entropy: "discrete" or "knn"
    series_names : list of str, optional
        Names for each series (default: "Series_0", "Series_1", ...)
    n_jobs : int, default 1
        Parallel workers for pairwise transfer entropy (-1 = all CPUs).
    
    Returns
    -------
    G : networkx.DiGraph
        Directed network with transfer entropy as edge weights
    te_matrix : array (n_series, n_series)
        Transfer entropy matrix (TE[i, j] = TE from series i to series j)
    stats : dict
        Network statistics including mean TE, max TE, etc.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = [np.random.randn(1000) for _ in range(5)]
    >>> G, te_matrix, stats = transfer_entropy_network(X, lag=1, threshold=0.1)
    >>> print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    >>> print(f"Mean transfer entropy: {stats['mean_te']:.4f} bits")
    """
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = [X]
        elif X.ndim == 2:
            X = [X[i] for i in range(X.shape[0])]
        else:
            raise ValueError(f"X must be 1D or 2D array, got shape {X.shape}")
    
    n_series = len(X)
    series_names = series_names or [f"Series_{i}" for i in range(n_series)]
    
    if len(series_names) != n_series:
        raise ValueError(
            f"series_names length ({len(series_names)}) must match "
            f"number of series ({n_series})"
        )
    
    def _te_pair(i: int, j: int) -> float:
        return transfer_entropy(X[i], X[j], lag=lag, bins=bins, method=method)

    pair_te = pairwise_parallel(n_series, _te_pair, n_jobs=n_jobs)

    te_matrix = np.zeros((n_series, n_series))
    for (i, j), te_val in pair_te.items():
        te_matrix[i, j] = te_val
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n_series))
    
    for i in range(n_series):
        for j in range(n_series):
            if i != j:
                te_val = te_matrix[i, j]
                if threshold is None or te_val >= threshold:
                    G.add_edge(i, j, weight=te_val)
    
    for i, name in enumerate(series_names):
        G.nodes[i]['name'] = name
    
    stats = {
        'mean_te': float(np.mean(te_matrix[te_matrix > 0])),
        'max_te': float(np.max(te_matrix)),
        'min_te': float(np.min(te_matrix[te_matrix > 0])) if np.any(te_matrix > 0) else 0.0,
        'std_te': float(np.std(te_matrix[te_matrix > 0])),
        'n_edges': G.number_of_edges(),
        'density': G.number_of_edges() / (n_series * (n_series - 1)) if n_series > 1 else 0.0,
    }
    
    return G, te_matrix, stats

