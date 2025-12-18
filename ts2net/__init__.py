"""
ts2net: Time Series to Networks

Clean API inspired by ts2vg, extended for multiple network methods.
"""

from .core.graph import Graph
from .api import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network

__version__ = "0.2.0"

__all__ = [
    'Graph',
    'HVG',
    'NVG',
    'RecurrenceNetwork',
    'TransitionNetwork',
    'build_network',
]
