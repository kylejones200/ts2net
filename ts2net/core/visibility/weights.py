"""
Weight computation utilities for visibility graphs.

Provides functions to compute edge weights using different modes:
- absdiff: Absolute difference in values
- time_gap: Temporal distance
- slope: Slope of the line connecting points
- min_clearance: Minimum clearance above intermediate points
"""

from __future__ import annotations

from typing import Literal
import numpy as np

WeightMode = Literal["absdiff", "time_gap", "min_clearance", "slope"]


def compute_weight(
    x: np.ndarray,
    i: int,
    j: int,
    mode: WeightMode
) -> float:
    """Compute edge weight for visibility graph edge (i, j).
    
    Parameters
    ----------
    x : array
        Time series values
    i : int
        Source node index
    j : int
        Destination node index
    mode : str
        Weight mode: "absdiff", "time_gap", "slope", or "min_clearance"
    
    Returns
    -------
    weight : float
        Edge weight
    """
    if i == j:
        return 0.0
    
    if mode == "absdiff":
        return float(abs(x[j] - x[i]))
    
    if mode == "time_gap":
        return float(abs(j - i))
    
    if mode == "slope":
        return float((x[j] - x[i]) / (j - i))
    
    if mode == "min_clearance":
        return float(_min_clearance(x, i=i, j=j))
    
    raise ValueError(f"Unknown weight mode: {mode}")


def _min_clearance(x: np.ndarray, *, i: int, j: int) -> float:
    """Compute minimum clearance between points i and j for visibility.
    
    The minimum clearance is the minimum distance between the baseline
    (min(x[i], x[j])) and the highest intermediate point.
    
    Parameters
    ----------
    x : array
        Time series values
    i : int
        First point index
    j : int
        Second point index
    
    Returns
    -------
    clearance : float
        Minimum clearance (inf if no intermediate points or all below baseline)
    """
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return float("inf")
    
    baseline = min(x[lo], x[hi])
    mid = x[lo + 1 : hi]
    if len(mid) == 0:
        return float("inf")
    
    return float(baseline - np.max(mid))


def recompute_weights(
    x: np.ndarray,
    edges: list,
    mode: WeightMode
) -> list:
    """Recompute weights for existing edges using a different weight mode.
    
    Parameters
    ----------
    x : array
        Time series values
    edges : list
        List of edges as (i, j) or (i, j, old_weight) tuples
    mode : str
        Weight mode to use
    
    Returns
    -------
    weighted_edges : list
        List of edges as (i, j, weight) tuples
    """
    weighted_edges = []
    for edge in edges:
        if len(edge) == 2:
            i, j = edge
        else:
            i, j = edge[0], edge[1]
        
        weight = compute_weight(x, i, j, mode)
        weighted_edges.append((i, j, weight))
    
    return weighted_edges
