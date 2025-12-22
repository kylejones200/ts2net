"""
Extract features from time series with optional BSTS decomposition.

Combines structural decomposition with network analysis of residuals.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

from .decompose import BSTSSpec, decompose, DecompositionResult
from ..api import HVG, NVG, TransitionNetwork


@dataclass
class FeaturesResult:
    """Result of feature extraction with BSTS decomposition.
    
    Attributes
    ----------
    raw_stats : dict
        Basic statistics of raw series (mean, std, min, max, etc.)
    structural_stats : dict
        Structural component statistics (variances, seasonal strength, etc.)
    residual_network_stats : dict
        Network statistics computed on residual (HVG, NVG, transition)
    """
    raw_stats: Dict[str, Any]
    structural_stats: Dict[str, Any]
    residual_network_stats: Dict[str, Any]


def _compute_raw_stats(series: NDArray[np.float64]) -> Dict[str, float]:
    """Compute basic statistics of raw series."""
    return {
        'mean': float(np.mean(series)),
        'std': float(np.std(series)),
        'min': float(np.min(series)),
        'max': float(np.max(series)),
        'median': float(np.median(series)),
        'q25': float(np.percentile(series, 25)),
        'q75': float(np.percentile(series, 75)),
        'n_points': len(series)
    }


def _compute_structural_stats(decomp: DecompositionResult) -> Dict[str, Any]:
    """Compute statistics of structural components."""
    stats = {
        'level_variance': decomp.level_variance,
        'error_variance': decomp.error_variance,
        'residual_std': float(np.std(decomp.residual)),
        'residual_mean': float(np.mean(decomp.residual)),
    }
    
    # Lag-1 autocorrelation of residual (whiteness proxy)
    if len(decomp.residual) > 1:
        lag1_corr = np.corrcoef(decomp.residual[:-1], decomp.residual[1:])[0, 1]
        stats['residual_lag1_autocorr'] = float(lag1_corr) if not np.isnan(lag1_corr) else 0.0
    else:
        stats['residual_lag1_autocorr'] = 0.0
    
    if decomp.trend_variance is not None:
        stats['trend_variance'] = decomp.trend_variance
    
    if decomp.seasonal_variance:
        stats['seasonal_variance'] = decomp.seasonal_variance
        # Compute seasonal strength (variance ratio)
        total_variance = decomp.level_variance + decomp.error_variance
        if decomp.trend_variance:
            total_variance += decomp.trend_variance
        seasonal_var_sum = sum(decomp.seasonal_variance.values())
        if total_variance > 0:
            stats['seasonal_strength'] = float(seasonal_var_sum / total_variance)
        else:
            stats['seasonal_strength'] = 0.0
    
    return stats


def _compute_network_stats(
    residual: NDArray[np.float64],
    methods: List[str],
    nvg_limit: Optional[int] = None
) -> Dict[str, Any]:
    """Compute network statistics on residual."""
    stats = {}
    
    if 'hvg' in methods:
        try:
            hvg = HVG(output='stats')
            hvg.build(residual)
            hvg_stats = hvg.stats()
            stats['hvg'] = {
                'n_nodes': hvg_stats['n_nodes'],
                'n_edges': hvg_stats['n_edges'],
                'avg_degree': hvg_stats['avg_degree'],
                'density': hvg_stats['density']
            }
        except Exception as e:
            stats['hvg_error'] = str(e)
    
    if 'nvg' in methods:
        try:
            nvg = NVG(output='stats', limit=nvg_limit or 3000)
            nvg.build(residual)
            nvg_stats = nvg.stats()
            stats['nvg'] = {
                'n_nodes': nvg_stats['n_nodes'],
                'n_edges': nvg_stats['n_edges'],
                'avg_degree': nvg_stats['avg_degree'],
                'density': nvg_stats['density']
            }
        except Exception as e:
            stats['nvg_error'] = str(e)
    
    if 'transition' in methods:
        try:
            tn = TransitionNetwork(symbolizer='ordinal', order=3, output='stats')
            tn.build(residual)
            tn_stats = tn.stats()
            stats['transition'] = {
                'n_nodes': tn_stats['n_nodes'],
                'n_edges': tn_stats['n_edges'],
                'avg_degree': tn_stats['avg_degree']
            }
        except Exception as e:
            stats['transition_error'] = str(e)
    
    return stats


def features(
    series: NDArray[np.float64],
    methods: List[str] = None,
    bsts: Optional[BSTSSpec] = None,
    window: Optional[int] = None,
    nvg_limit: Optional[int] = None
) -> FeaturesResult:
    """
    Extract features from time series with optional BSTS decomposition.
    
    If BSTS is enabled, decomposes series and analyzes residual with network methods.
    If BSTS is disabled, analyzes raw series.
    
    Parameters
    ----------
    series : array
        Input time series
    methods : list of str, optional
        Network methods to apply: 'hvg', 'nvg', 'transition'
        Default: ['hvg', 'transition']
    bsts : BSTSSpec, optional
        BSTS decomposition specification. If None, analyzes raw series.
    window : int, optional
        Window size for windowed analysis. If None, analyzes full series.
    nvg_limit : int, optional
        Horizon limit for NVG (default: 3000)
    
    Returns
    -------
    FeaturesResult
        Three blocks: raw_stats, structural_stats, residual_network_stats
    
    Examples
    --------
    >>> from ts2net.bsts import features, BSTSSpec
    >>> spec = BSTSSpec(level=True, seasonal_periods=[24, 168])
    >>> result = features(x, methods=['hvg', 'transition'], bsts=spec)
    >>> print(result.structural_stats['seasonal_strength'])
    >>> print(result.residual_network_stats['hvg']['avg_degree'])
    """
    if methods is None:
        methods = ['hvg', 'transition']
    
    # Compute raw statistics
    raw_stats = _compute_raw_stats(series)
    
    # If no BSTS, analyze raw series
    if bsts is None:
        network_stats = _compute_network_stats(series, methods, nvg_limit)
        return FeaturesResult(
            raw_stats=raw_stats,
            structural_stats={},
            residual_network_stats=network_stats
        )
    
    # Decompose with BSTS
    try:
        decomp = decompose(series, bsts)
    except Exception as e:
        # If decomposition fails, return raw stats and error
        return FeaturesResult(
            raw_stats=raw_stats,
            structural_stats={'decomposition_error': str(e)},
            residual_network_stats={}
        )
    
    # Compute structural statistics
    structural_stats = _compute_structural_stats(decomp)
    
    # Analyze residual with network methods
    if window is None:
        # Analyze full residual
        residual_network_stats = _compute_network_stats(decomp.residual, methods, nvg_limit)
    else:
        # Windowed analysis - aggregate across windows
        window_stats = []
        for i in range(0, len(decomp.residual) - window + 1, window):
            window_residual = decomp.residual[i:i + window]
            if np.std(window_residual) > 0:  # Skip constant windows
                window_stat = _compute_network_stats(window_residual, methods, nvg_limit)
                window_stats.append(window_stat)
        
        # Aggregate window statistics (median, IQR)
        if window_stats:
            residual_network_stats = {}
            for method in methods:
                if method in window_stats[0]:
                    method_stats = [ws[method] for ws in window_stats if method in ws]
                    if method_stats:
                        # Aggregate key metrics
                        avg_degrees = [ms['avg_degree'] for ms in method_stats]
                        n_edges = [ms['n_edges'] for ms in method_stats]
                        residual_network_stats[method] = {
                            'avg_degree_median': float(np.median(avg_degrees)),
                            'avg_degree_q25': float(np.percentile(avg_degrees, 25)),
                            'avg_degree_q75': float(np.percentile(avg_degrees, 75)),
                            'n_edges_median': float(np.median(n_edges)),
                            'n_windows': len(method_stats)
                        }
        else:
            residual_network_stats = {}
    
    return FeaturesResult(
        raw_stats=raw_stats,
        structural_stats=structural_stats,
        residual_network_stats=residual_network_stats
    )
