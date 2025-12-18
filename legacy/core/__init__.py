"""
Core module for time series to network conversion.

This module provides implementations of various time series to network conversion
algorithms including Recurrence Networks, Visibility Graphs, and Transition Networks.
"""

from .recurrence import RecurrenceNetwork
from .transition import TransitionNetwork
from .visibility import HVG, NVG
from .parallel import batch_transform, _run_single

# For backward compatibility
__all__ = [
    "RecurrenceNetwork",
    "TransitionNetwork",
    "HVG",
    "NVG",
    "batch_transform",
    "_run_single",
]
