"""
Hard correctness tests for NVG implementation.

Tests compare fast implementation against naive O(n²) reference on small series.
"""

import numpy as np
import pytest
from ts2net.core.visibility.nvg import _nvg_edges_numba
from ts2net.api import NVG


def nvg_edges_naive(x: np.ndarray, weighted: bool = False) -> set:
    """
    Naive O(n²) reference implementation for NVG edge computation.
    
    For each pair (i, j) with i < j, check if the line connecting (i, x[i])
    and (j, x[j]) does not intersect any intermediate points.
    
    Returns set of edges as (min(i,j), max(i,j)) tuples.
    """
    n = len(x)
    edges = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check natural visibility
            # Line from (i, x[i]) to (j, x[j])
            # For each k between i and j, check if point (k, x[k]) is above the line
            visible = True
            for k in range(i + 1, j):
                # Line equation: y = x[i] + slope * (k - i)
                # where slope = (x[j] - x[i]) / (j - i)
                line_height = x[i] + (x[j] - x[i]) * (k - i) / (j - i)
                if x[k] > line_height:
                    visible = False
                    break
            
            if visible:
                edges.add((i, j))
    
    return edges


def normalize_edges(rows: np.ndarray, cols: np.ndarray) -> set:
    """Normalize edges to canonical form (min, max) for comparison."""
    return {(min(i, j), max(i, j)) for i, j in zip(rows, cols)}


class TestNVGCorrectness:
    """Test NVG correctness against naive reference."""
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_random_series(self, n):
        """Test random series of various lengths."""
        np.random.seed(42)
        x = np.random.randn(n)
        
        # Fast implementation (no limit for correctness test)
        rows, cols, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        # Naive reference
        edges_naive = nvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for random series n={n}: {len(edges_fast)} vs {len(edges_naive)}"
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_monotone_increasing(self, n):
        """Test strictly increasing series."""
        x = np.linspace(0, 1, n)
        
        rows, cols, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = nvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for monotone increasing n={n}"
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_periodic_series(self, n):
        """Test periodic series (sine wave)."""
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        
        rows, cols, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = nvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for periodic series n={n}"
    
    @pytest.mark.parametrize("seed", range(10))
    def test_multiple_random_seeds(self, seed):
        """Test multiple random seeds to catch edge cases."""
        np.random.seed(seed)
        n = 150
        x = np.random.randn(n)
        
        rows, cols, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = nvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for seed={seed}"


class TestNVGInvariants:
    """Test mathematical invariants that must hold for NVG."""
    
    def test_invariant_to_constant_add(self):
        """NVG must be invariant to adding a constant."""
        np.random.seed(42)
        x = np.random.randn(100)
        x_shifted = x + 10.0
        
        rows1, cols1, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        rows2, cols2, _, _ = _nvg_edges_numba(x_shifted, weighted=False, limit=-1)
        
        edges1 = normalize_edges(rows1, cols1)
        edges2 = normalize_edges(rows2, cols2)
        
        assert edges1 == edges2, "NVG should be invariant to adding a constant"
    
    def test_invariant_to_positive_scaling(self):
        """NVG must be invariant to multiplying by positive constant."""
        np.random.seed(42)
        x = np.random.randn(100)
        x_scaled = 3.0 * x
        
        rows1, cols1, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        rows2, cols2, _, _ = _nvg_edges_numba(x_scaled, weighted=False, limit=-1)
        
        edges1 = normalize_edges(rows1, cols1)
        edges2 = normalize_edges(rows2, cols2)
        
        assert edges1 == edges2, "NVG should be invariant to positive scaling"
    
    def test_consecutive_edges_lower_bound(self):
        """NVG must include edges between consecutive points."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        
        rows, cols, _, _ = _nvg_edges_numba(x, weighted=False, limit=-1)
        edges = normalize_edges(rows, cols)
        
        # All consecutive pairs (i, i+1) for i in [0, n-2] must be edges
        consecutive_edges = {(i, i + 1) for i in range(n - 1)}
        
        assert consecutive_edges.issubset(edges), \
            "NVG must include all consecutive edges"
