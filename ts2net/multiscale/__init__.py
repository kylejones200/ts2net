"""
Multiscale graph analysis for time series.

Provides functionality to analyze time series at multiple temporal scales
by coarse-graining and computing graph features at each scale.
"""

from .core import MultiscaleGraphs, coarse_grain

__all__ = ["MultiscaleGraphs", "coarse_grain"]


