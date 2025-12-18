"""
Distance metrics for time series analysis.

This module provides various distance metrics for comparing time series,
including correlation-based, dynamic time warping, and information-theoretic measures.
"""

from .core import (
    tsdist_cor,
    tsdist_ccf,
    tsdist_dtw,
    tsdist_nmi,
    tsdist_voi,
    tsdist_mic,
    tsdist_vr,
    dist_percentile,
    dist_matrix_normalize,
)

__all__ = [
    "tsdist_cor",
    "tsdist_ccf",
    "tsdist_dtw",
    "tsdist_nmi",
    "tsdist_voi",
    "tsdist_mic",
    "tsdist_vr",
    "dist_percentile",
    "dist_matrix_normalize",
]
