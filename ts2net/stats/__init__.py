"""
Statistical functions for time series and network analysis.

This module provides statistical functions for analyzing time series data
and networks, including correlation analysis, surrogate data generation,
recurrence analysis, and network motif analysis.
"""

from .stats import (
    permutation_entropy,
    cao_e1_e2,
    false_nearest_neighbors,
    rqa_full,
    corr_sig,
    ccf_sig,
    fisher_z_ci,
    surrogate_phase,
    surrogate_circular,
    iaaft,
    motif_zscore,
    rqa_measures,
    partial_corr,
    partial_corr_sig,
)

from .null_models import (
    generate_surrogate,
    compute_network_metric_significance,
    compute_multiple_metrics_significance,
    compute_zscore,
    NetworkSignificanceResult,
)

__all__ = [
    'permutation_entropy',
    'cao_e1_e2', 
    'false_nearest_neighbors',
    'rqa_full',
    'corr_sig',
    'ccf_sig',
    'fisher_z_ci',
    'surrogate_phase',
    'surrogate_circular',
    'iaaft',
    'motif_zscore',
    'rqa_measures',
    'partial_corr',
    'partial_corr_sig',
    # Null models and significance testing
    'generate_surrogate',
    'compute_network_metric_significance',
    'compute_multiple_metrics_significance',
    'compute_zscore',
    'NetworkSignificanceResult',
]
