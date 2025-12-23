"""
Minimal invariant tests for graph implementations.

Tests only that graphs build without error and satisfy basic invariants.
"""

import numpy as np
import pytest
from ts2net.core.visibility import HVG, NVG


def test_hvg_builds():
    """HVG builds without error and satisfies invariants."""
    ts = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
    hvg = HVG(weighted=True)
    G, A = hvg.fit_transform(ts)
    
    # Invariants: node count matches input length
    assert G.number_of_nodes() == len(ts)
    # Edge count within bounds (at least consecutive, at most complete)
    assert G.number_of_edges() >= len(ts) - 1
    assert G.number_of_edges() <= len(ts) * (len(ts) - 1) // 2


def test_nvg_builds():
    """NVG builds without error and satisfies invariants."""
    ts = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
    nvg = NVG(weighted=True)
    G, A = nvg.fit_transform(ts)
    
    # Invariants: node count matches input length
    assert G.number_of_nodes() == len(ts)
    # Edge count within bounds
    assert G.number_of_edges() >= len(ts) - 1
    assert G.number_of_edges() <= len(ts) * (len(ts) - 1) // 2
