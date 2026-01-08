"""
Feature-wise network comparisons for multiple time series.

This module provides functions to compute network statistics for multiple
time series and compare them, enabling analysis of how different series
behave in terms of their network properties.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Union, Literal
import warnings
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix

from ..api import HVG, NVG
from ..core.recurrence import RecurrenceNetwork
from ..core.transition import TransitionNetwork


def _extract_graph_stats(G: Union[nx.Graph, csr_matrix]) -> Dict[str, float]:
    """Extract statistics from graph (NetworkX or sparse matrix)."""
    if isinstance(G, nx.Graph):
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        degrees = np.array([d for _, d in G.degree()])
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": nx.density(G),
            "avg_degree": float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
            "std_degree": float(np.std(degrees)) if len(degrees) > 1 else 0.0,
            "min_degree": float(np.min(degrees)) if len(degrees) > 0 else 0.0,
            "max_degree": float(np.max(degrees)) if len(degrees) > 0 else 0.0,
        }
    n_nodes = G.shape[0]
    n_edges = G.nnz // 2
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
    }


def _build_network_stats(method: str, x: NDArray[np.float64], **kwargs) -> Dict[str, float]:
    """Build network and extract statistics using method dispatch."""
    method_builders = {
        "hvg": lambda: HVG(**kwargs),
        "nvg": lambda: NVG(**kwargs),
    }
    
    method_transformers = {
        "recurrence": lambda: RecurrenceNetwork(**kwargs),
        "transition": lambda: TransitionNetwork(**kwargs),
    }
    
    if method in method_builders:
        builder = method_builders[method]()
        builder.build(x)
        return builder.stats()
    
    if method in method_transformers:
        builder = method_transformers[method]()
        G = builder.fit(x).transform()
        return _extract_graph_stats(G)
    
    raise ValueError(f"Unknown method: {method}. Use 'hvg', 'nvg', 'recurrence', or 'transition'")


def compute_network_features(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    method: Literal["hvg", "nvg", "recurrence", "transition"] = "hvg",
    series_names: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compute network features for multiple time series.
    
    For each time series, builds a network and extracts summary statistics.
    Returns a DataFrame with one row per series and columns for each feature.
    
    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Multiple time series to analyze
    method : str, default "hvg"
        Network construction method: "hvg", "nvg", "recurrence", or "transition"
    series_names : list of str, optional
        Names for each series (default: "Series_0", "Series_1", ...)
    **kwargs
        Additional arguments passed to network builder (e.g., weighted, k, threshold)
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with network features for each series:
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - density: Edge density
        - avg_degree: Average degree
        - std_degree: Standard deviation of degree
        - min_degree: Minimum degree
        - max_degree: Maximum degree
        - (and method-specific features)
    
    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.multivariate import compute_network_features
    >>> 
    >>> # Create multiple time series
    >>> X = [np.random.randn(100) for _ in range(5)]
    >>> 
    >>> # Compute HVG features for all series
    >>> features = compute_network_features(X, method="hvg")
    >>> print(features)
    >>> 
    >>> # Compare series
    >>> print(features.describe())
    """
    # Normalize input format
    if isinstance(X, np.ndarray):
        X = [X] if X.ndim == 1 else [X[i] for i in range(X.shape[0])]
        if X[0].ndim > 1:
            raise ValueError(f"X must be 1D or 2D array, got shape {X[0].shape}")
    
    n_series = len(X)
    series_names = series_names or [f"Series_{i}" for i in range(n_series)]
    
    if len(series_names) != n_series:
        raise ValueError(f"series_names length ({len(series_names)}) must match number of series ({n_series})")
    
    # Build networks and extract features
    all_features = []
    default_stats = {
        "n_nodes": np.nan,
        "n_edges": np.nan,
        "density": np.nan,
        "avg_degree": np.nan,
        "std_degree": np.nan,
        "min_degree": np.nan,
        "max_degree": np.nan,
    }
    
    for i, x in enumerate(X):
        try:
            stats = _build_network_stats(method, x, **kwargs)
            stats["series_name"] = series_names[i]
            all_features.append(stats)
        except ValueError:
            raise
        except Exception as e:
            warnings.warn(f"Failed to process series {i} ({series_names[i]}): {e}")
            stats = default_stats.copy()
            stats["series_name"] = series_names[i]
            all_features.append(stats)
    
    df = pd.DataFrame(all_features)
    if "series_name" in df.columns:
        df = df.set_index("series_name")
    
    return df


def compare_network_features(
    features_df: pd.DataFrame,
    metric: Optional[str] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compare network features across multiple series.
    
    Computes summary statistics and similarity measures for network features
    across different time series.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame from `compute_network_features()` with network features
        for multiple series
    metric : str, optional
        Specific metric to compare (if None, compares all numeric columns)
    
    Returns
    -------
    comparison : dict
        Dictionary with comparison metrics:
        - "mean": Mean value across series
        - "std": Standard deviation across series
        - "min": Minimum value
        - "max": Maximum value
        - "range": Range (max - min)
        - "cv": Coefficient of variation (std / mean)
        - "similarity_matrix": Correlation matrix of features (if multiple metrics)
    
    Examples
    --------
    >>> features = compute_network_features(X, method="hvg")
    >>> comparison = compare_network_features(features)
    >>> print(f"Avg density: {comparison['density']['mean']:.3f}")
    >>> print(f"Density CV: {comparison['density']['cv']:.3f}")
    """
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {"error": "No numeric features found"}
    
    if metric is not None:
        if metric not in numeric_cols:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(numeric_cols)}")
        numeric_cols = [metric]
    
    comparison = {}
    
    for col in numeric_cols:
        values = features_df[col].dropna()
        if len(values) == 0:
            comparison[col] = {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "range": np.nan,
                "cv": np.nan,
            }
        else:
            mean_val = float(values.mean())
            std_val = float(values.std())
            min_val = float(values.min())
            max_val = float(values.max())
            
            comparison[col] = {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "cv": std_val / mean_val if mean_val != 0 else np.nan,
            }
    
    if len(numeric_cols) > 1:
        similarity = features_df[numeric_cols].T.corr()
        comparison["similarity_matrix"] = similarity.to_dict()
    
    return comparison


def cluster_series_by_features(
    features_df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    method: str = "kmeans"
) -> Dict[str, int]:
    """
    Cluster time series based on their network features.
    
    Groups series with similar network properties together.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame from `compute_network_features()` with network features
    n_clusters : int, optional
        Number of clusters (if None, uses elbow method)
    method : str, default "kmeans"
        Clustering method: "kmeans" or "hierarchical"
    
    Returns
    -------
    clusters : dict
        Dictionary mapping series name to cluster ID
    
    Examples
    --------
    >>> features = compute_network_features(X, method="hvg")
    >>> clusters = cluster_series_by_features(features, n_clusters=3)
    >>> print(f"Series grouped into {len(set(clusters.values()))} clusters")
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    X = features_df[numeric_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = n_clusters or max(2, int(np.sqrt(len(features_df) / 2)))
    
    clusterers = {
        "kmeans": lambda: KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "hierarchical": lambda: AgglomerativeClustering(n_clusters=n_clusters),
    }
    
    if method not in clusterers:
        raise ValueError(f"Unknown method: {method}")
    
    clusterer = clusterers[method]()
    labels = clusterer.fit_predict(X_scaled)
    
    return {name: int(label) for name, label in zip(features_df.index, labels)}
