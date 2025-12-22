"""
Bayesian Structural Time Series (BSTS) decomposition and residual topology analysis.

This module provides structural decomposition of time series and network analysis
of residuals to separate predictable structure from irregular dynamics.
"""

from .decompose import BSTSSpec, decompose
from .features import features, FeaturesResult

__all__ = ['BSTSSpec', 'decompose', 'features', 'FeaturesResult']
