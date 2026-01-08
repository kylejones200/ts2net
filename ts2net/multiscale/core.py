"""
Multiscale graph analysis core implementation.

Coarse-grains time series at multiple scales and computes graph features
at each scale to create a scale signature for detection stability.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Union
import numpy as np
from numpy.typing import NDArray

from ..factory import create_graph_builder
from ..config import HVGConfig, NVGConfig, RecurrenceConfig, TransitionConfig


def coarse_grain(x: NDArray[np.float64], scale: int, method: str = "mean") -> NDArray[np.float64]:
    """
    Coarse-grain a time series by aggregating points at a given scale.
    
    Parameters
    ----------
    x : array (n_points,)
        Input time series
    scale : int
        Coarse-graining scale (number of points to aggregate)
    method : str, default "mean"
        Aggregation method: "mean", "median", "max", "min", "std"
    
    Returns
    -------
    array (n_points // scale,)
        Coarse-grained time series
    
    Examples
    --------
    >>> x = np.arange(12.0)
    >>> coarse_grain(x, scale=3, method="mean")
    array([1., 4., 7., 10.])
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    
    if scale >= len(x):
        raise ValueError(f"scale ({scale}) must be less than series length ({len(x)})")
    
    n = len(x)
    n_coarse = n // scale
    
    # Truncate to multiple of scale
    x_truncated = x[:n_coarse * scale]
    x_reshaped = x_truncated.reshape(n_coarse, scale)
    
    if method == "mean":
        return np.mean(x_reshaped, axis=1)
    elif method == "median":
        return np.median(x_reshaped, axis=1)
    elif method == "max":
        return np.max(x_reshaped, axis=1)
    elif method == "min":
        return np.min(x_reshaped, axis=1)
    elif method == "std":
        return np.std(x_reshaped, axis=1, ddof=1)
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: mean, median, max, min, std")


class MultiscaleGraphs:
    """
    Multiscale graph analysis for time series.
    
    Analyzes time series at multiple temporal scales by coarse-graining
    and computing graph features at each scale. Creates a scale signature
    (feature vector across scales) useful for detection stability.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4, 8])
    >>> ms.fit(x)
    >>> signature = ms.scale_signature()
    >>> # signature is a dict with features at each scale
    >>> print(signature['avg_degree'])  # Array of avg_degree at each scale
    """
    
    def __init__(
        self,
        method: str = "hvg",
        scales: Optional[List[int]] = None,
        coarse_method: str = "mean",
        output: str = "stats",
        **method_kwargs
    ):
        """
        Initialize multiscale graph analyzer.
        
        Parameters
        ----------
        method : str, default "hvg"
            Network method: 'hvg', 'nvg', 'recurrence', 'transition'
        scales : list of int, optional
            List of coarse-graining scales. If None, uses [1, 2, 4, 8, 16]
        coarse_method : str, default "mean"
            Coarse-graining aggregation: "mean", "median", "max", "min", "std"
        output : str, default "stats"
            Output mode: 'stats', 'degrees', or 'edges'
        **method_kwargs
            Additional parameters for the network builder
        """
        self.method = method.lower()
        if scales is None:
            # Default scales: powers of 2 up to reasonable limit
            self.scales = [1, 2, 4, 8, 16]
        else:
            self.scales = sorted(scales)  # Sort for consistency
        self.coarse_method = coarse_method
        self.output = output
        self.method_kwargs = method_kwargs
        
        self.x_ = None
        self.scale_stats_ = None
    
    def fit(self, x: NDArray[np.float64]) -> 'MultiscaleGraphs':
        """
        Fit the multiscale analyzer to a time series.
        
        Parameters
        ----------
        x : array (n_points,)
            Input time series
        
        Returns
        -------
        self : MultiscaleGraphs
            Returns self for method chaining
        """
        x = np.asarray(x, dtype=np.float64).squeeze()
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")
        
        if len(x) < max(self.scales):
            raise ValueError(
                f"Series length ({len(x)}) must be >= max scale ({max(self.scales)})"
            )
        
        self.x_ = x
        self.scale_stats_ = {}
        
        # Create config factory based on method
        def _create_hvg_config():
            return HVGConfig(
                enabled=True,
                output=self.output,
                weighted=self.method_kwargs.get('weighted', False),
                weight_mode=self.method_kwargs.get('weight_mode'),
                limit=self.method_kwargs.get('limit'),
                directed=self.method_kwargs.get('directed', False)
            )
        
        def _create_nvg_config():
            return NVGConfig(
                enabled=True,
                output=self.output,
                weighted=self.method_kwargs.get('weighted', False),
                weight_mode=self.method_kwargs.get('weight_mode'),
                limit=self.method_kwargs.get('limit'),
                max_edges=self.method_kwargs.get('max_edges'),
                max_edges_per_node=self.method_kwargs.get('max_edges_per_node'),
                max_memory_mb=self.method_kwargs.get('max_memory_mb')
            )
        
        def _create_recurrence_config():
            return RecurrenceConfig(
                enabled=True,
                output=self.output,
                m=self.method_kwargs.get('m', 3),
                rule='knn',
                k=self.method_kwargs.get('k', 5),
                tau=self.method_kwargs.get('tau', 1),
                epsilon=self.method_kwargs.get('epsilon', 0.1),
                metric=self.method_kwargs.get('metric', 'euclidean')
            )
        
        def _create_transition_config():
            return TransitionConfig(
                enabled=True,
                output=self.output,
                symbolizer=self.method_kwargs.get('symbolizer', 'ordinal'),
                order=self.method_kwargs.get('order', 3),
                partition_mode=self.method_kwargs.get('partition_mode', False),
                n_states=self.method_kwargs.get('n_states')
            )
        
        config_map = {
            'hvg': _create_hvg_config,
            'nvg': _create_nvg_config,
            'recurrence': _create_recurrence_config,
            'transition': _create_transition_config,
        }
        
        config_factory = config_map.get(self.method)
        if config_factory is None:
            raise ValueError(f"Unknown method: {self.method}. Must be one of {list(config_map.keys())}")
        
        # Compute graph features at each scale
        for scale in self.scales:
            try:
                # Coarse-grain the series
                if scale == 1:
                    x_coarse = self.x_
                else:
                    x_coarse = coarse_grain(self.x_, scale=scale, method=self.coarse_method)
                
                # Skip if coarse-grained series is too short
                if len(x_coarse) < 10:
                    self.scale_stats_[scale] = None
                    continue
                
                # Build graph at this scale
                config = config_factory()
                builder = create_graph_builder(self.method, config, n_points=len(x_coarse))
                builder.build(x_coarse)
                stats = builder.stats(include_triangles=False)
                
                self.scale_stats_[scale] = stats
                
            except Exception as e:
                # Handle errors gracefully (e.g., constant series, too short)
                import warnings
                warnings.warn(f"Failed to compute graph at scale {scale}: {e}")
                self.scale_stats_[scale] = None
        
        return self
    
    def scale_signature(self, features: Optional[List[str]] = None) -> Dict[str, NDArray[np.float64]]:
        """
        Get scale signature (feature values across scales).
        
        Parameters
        ----------
        features : list of str, optional
            List of feature names to include. If None, uses common features:
            ['n_nodes', 'n_edges', 'avg_degree', 'std_degree', 'density']
        
        Returns
        -------
        dict[str, array]
            Dictionary mapping feature names to arrays of length n_scales.
            Each array contains the feature value at each scale.
        
        Examples
        --------
        >>> ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4])
        >>> ms.fit(x)
        >>> signature = ms.scale_signature()
        >>> print(signature['avg_degree'])  # [deg_scale1, deg_scale2, deg_scale4]
        """
        if self.scale_stats_ is None:
            raise ValueError("Must call fit() first")
        
        if features is None:
            features = ['n_nodes', 'n_edges', 'avg_degree', 'std_degree', 'density']
        
        signature = {}
        
        for feature in features:
            values = []
            for scale in self.scales:
                stats = self.scale_stats_.get(scale)
                if stats is None:
                    values.append(np.nan)
                else:
                    values.append(stats.get(feature, np.nan))
            signature[feature] = np.array(values, dtype=np.float64)
        
        return signature
    
    def stats(self) -> Dict[int, Dict]:
        """
        Get full statistics at each scale.
        
        Returns
        -------
        dict[int, dict]
            Dictionary mapping scale to statistics dictionary
        """
        if self.scale_stats_ is None:
            raise ValueError("Must call fit() first")
        
        return self.scale_stats_.copy()
    
    def fit_transform(self, x: NDArray[np.float64], features: Optional[List[str]] = None) -> Dict[str, NDArray[np.float64]]:
        """
        Fit and return scale signature in one step.
        
        Parameters
        ----------
        x : array (n_points,)
            Input time series
        features : list of str, optional
            Features to include in signature
        
        Returns
        -------
        dict[str, array]
            Scale signature dictionary
        """
        return self.fit(x).scale_signature(features=features)

