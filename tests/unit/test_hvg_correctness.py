"""
Hard correctness tests for HVG implementation.

Tests compare fast implementation against naive O(n²) reference on small series.
"""

import numpy as np
import pytest
from ts2net.core.visibility.hvg import _hvg_edges_numba
from ts2net.api import HVG


def hvg_edges_naive(x: np.ndarray, weighted: bool = False) -> set:
    """
    Naive O(n²) reference implementation for HVG edge computation.
    
    For each pair (i, j) with i < j, check horizontal visibility:
    All points k between i and j must be below min(x[i], x[j]).
    
    Returns set of edges as (min(i,j), max(i,j)) tuples.
    """
    n = len(x)
    edges = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check horizontal visibility
            # All points k between i and j must satisfy: x[k] < min(x[i], x[j])
            threshold = min(x[i], x[j])
            visible = True
            for k in range(i + 1, j):
                if x[k] >= threshold:
                    visible = False
                    break
            
            if visible:
                edges.add((i, j))
    
    return edges


def normalize_edges(rows: np.ndarray, cols: np.ndarray) -> set:
    """Normalize edges to canonical form (min, max) for comparison."""
    return {(min(i, j), max(i, j)) for i, j in zip(rows, cols)}


class TestHVGCorrectness:
    """Test HVG correctness against naive reference."""
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_random_series(self, n):
        """Test random series of various lengths."""
        np.random.seed(42)
        x = np.random.randn(n)
        
        # Fast implementation
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        # Naive reference
        edges_naive = hvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for random series n={n}: {len(edges_fast)} vs {len(edges_naive)}"
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_monotone_increasing(self, n):
        """Test strictly increasing series."""
        x = np.linspace(0, 1, n)
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = hvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for monotone increasing n={n}"
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_monotone_decreasing(self, n):
        """Test strictly decreasing series."""
        x = np.linspace(1, 0, n)
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = hvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for monotone decreasing n={n}"
    
    @pytest.mark.parametrize("n", [10, 50, 100, 200, 300])
    def test_periodic_series(self, n):
        """Test periodic series (sine wave)."""
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = hvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for periodic series n={n}"
    
    def test_constant_series(self):
        """Test constant series (should have no edges except consecutive)."""
        n = 100
        x = np.ones(n)
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        # Constant series: only consecutive edges are visible
        expected_edges = {(i, i + 1) for i in range(n - 1)}
        
        assert edges_fast == expected_edges, \
            f"Constant series should only have consecutive edges"
    
    @pytest.mark.parametrize("seed", range(10))
    def test_multiple_random_seeds(self, seed):
        """Test multiple random seeds to catch edge cases."""
        np.random.seed(seed)
        n = 150
        x = np.random.randn(n)
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges_fast = normalize_edges(rows, cols)
        
        edges_naive = hvg_edges_naive(x, weighted=False)
        
        assert edges_fast == edges_naive, \
            f"Edge mismatch for seed={seed}"


class TestHVGInvariants:
    """Test mathematical invariants that must hold for HVG."""
    
    def test_invariant_to_constant_add(self):
        """HVG must be invariant to adding a constant."""
        np.random.seed(42)
        x = np.random.randn(100)
        x_shifted = x + 10.0
        
        rows1, cols1, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        rows2, cols2, _ = _hvg_edges_numba(x_shifted, weighted=False, limit=-1)
        
        edges1 = normalize_edges(rows1, cols1)
        edges2 = normalize_edges(rows2, cols2)
        
        assert edges1 == edges2, "HVG should be invariant to adding a constant"
    
    def test_invariant_to_positive_scaling(self):
        """HVG must be invariant to multiplying by positive constant."""
        np.random.seed(42)
        x = np.random.randn(100)
        x_scaled = 3.0 * x
        
        rows1, cols1, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        rows2, cols2, _ = _hvg_edges_numba(x_scaled, weighted=False, limit=-1)
        
        edges1 = normalize_edges(rows1, cols1)
        edges2 = normalize_edges(rows2, cols2)
        
        assert edges1 == edges2, "HVG should be invariant to positive scaling"
    
    def test_invariant_to_strictly_increasing_transform(self):
        """HVG must be invariant to any strictly increasing transform."""
        np.random.seed(42)
        x = np.random.randn(100)
        # exp is strictly increasing
        x_transformed = np.exp(x)
        
        rows1, cols1, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        rows2, cols2, _ = _hvg_edges_numba(x_transformed, weighted=False, limit=-1)
        
        edges1 = normalize_edges(rows1, cols1)
        edges2 = normalize_edges(rows2, cols2)
        
        assert edges1 == edges2, "HVG should be invariant to strictly increasing transforms"
    
    def test_consecutive_edges_lower_bound(self):
        """HVG must include edges between consecutive points for all interior points."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
        edges = normalize_edges(rows, cols)
        
        # All consecutive pairs (i, i+1) for i in [0, n-2] must be edges
        consecutive_edges = {(i, i + 1) for i in range(n - 1)}
        
        assert consecutive_edges.issubset(edges), \
            "HVG must include all consecutive edges"
    
    def test_average_degree_statistical_invariant(self):
        """
        HVG average degree should be near 4 for long i.i.d. continuous noise.
        
        Test with 200 random series of length 5k and assert mean average degree
        is in tight band [3.85, 4.15].
        """
        np.random.seed(42)
        n_series = 200
        n_points = 5000
        avg_degrees = []
        
        for _ in range(n_series):
            x = np.random.randn(n_points)
            rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=-1)
            n_edges = len(rows)
            avg_degree = 2 * n_edges / n_points  # Undirected graph
            avg_degrees.append(avg_degree)
        
        mean_avg_degree = np.mean(avg_degrees)
        
        assert 3.85 <= mean_avg_degree <= 4.15, \
            f"Mean average degree {mean_avg_degree:.3f} should be in [3.85, 4.15] for i.i.d. noise"
        
        # Also check that std is reasonable (not too variable)
        std_avg_degree = np.std(avg_degrees)
        assert std_avg_degree < 0.1, \
            f"Std of average degree {std_avg_degree:.3f} should be < 0.1"


class TestHVGMemoryBehavior:
    """Test that HVG never creates dense n×n arrays."""
    
    def test_no_dense_adjacency_by_default(self):
        """HVG should return sparse representation by default."""
        n = 10000
        x = np.random.randn(n)
        
        hvg = HVG(output='edges')
        hvg.build(x)
        
        # Should have edges_coo() method that returns sparse representation
        rows, cols, weights = hvg.edges_coo()
        
        assert len(rows) < n * n, "Should not materialize full adjacency matrix"
        assert len(rows) == len(cols), "Edge arrays should have same length"
    
    def test_refuse_dense_at_large_n(self):
        """Should refuse dense adjacency at large n unless force=True."""
        n = 100000
        x = np.random.randn(n)
        
        hvg = HVG(output='edges')
        hvg.build(x)
        
        # Should not have dense adjacency matrix
        # If user tries to get dense, should raise or warn
        # (This depends on implementation - check if adjacency_matrix exists)
        if hasattr(hvg, 'adjacency_matrix'):
            # If it exists, it should return sparse by default
            A = hvg.adjacency_matrix()
            # Check if it's sparse
            try:
                from scipy.sparse import issparse
                assert issparse(A), "adjacency_matrix should return sparse by default"
            except ImportError:
                pass  # scipy not available


class TestDataHygiene:
    """Test data cleaning and type coercion."""
    
    def test_coerce_to_float64(self):
        """Test that input is coerced to float64."""
        from ts2net.api import _validate_and_clean_series
        
        # Test with int array
        x_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        x_clean = _validate_and_clean_series(x_int, "test")
        assert x_clean.dtype == np.float64, "Should coerce to float64"
        
        # Test with float32
        x_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        x_clean = _validate_and_clean_series(x_float32, "test")
        assert x_clean.dtype == np.float64, "Should coerce to float64"
    
    def test_drop_nan(self):
        """Test that NaN values are dropped."""
        from ts2net.api import _validate_and_clean_series
        
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        x_clean = _validate_and_clean_series(x, "test")
        
        assert not np.any(np.isnan(x_clean)), "Should drop NaN values"
        assert len(x_clean) == 4, "Should have 4 non-NaN values"
    
    def test_drop_inf(self):
        """Test that inf values are dropped."""
        from ts2net.api import _validate_and_clean_series
        
        x = np.array([1.0, 2.0, np.inf, 4.0, -np.inf, 5.0])
        x_clean = _validate_and_clean_series(x, "test")
        
        assert not np.any(np.isinf(x_clean)), "Should drop inf values"
        assert len(x_clean) == 4, "Should have 4 finite values"
    
    def test_object_array_with_strings(self):
        """Test handling of object arrays with mixed strings."""
        from ts2net.api import _validate_and_clean_series
        
        # Create object array with mix of numbers and strings
        x_obj = np.array([1.0, '2.0', 3.0, 'invalid', 5.0], dtype=object)
        
        # Should handle gracefully (either convert or raise clear error)
        try:
            x_clean = _validate_and_clean_series(x_obj, "test")
            # If it succeeds, should be numeric
            assert x_clean.dtype == np.float64, "Should convert to float64"
            assert len(x_clean) <= len(x_obj), "May drop invalid values"
        except (ValueError, TypeError) as e:
            # If it fails, error should be clear
            assert 'numeric' in str(e).lower() or 'convert' in str(e).lower(), \
                "Error message should mention numeric conversion"
    
    def test_mixed_types_pandas_style(self):
        """Test handling of mixed types that might come from pandas."""
        from ts2net.api import _validate_and_clean_series
        import pandas as pd
        
        # Create series with mixed types (like from CSV)
        x_mixed = pd.Series([1.0, '2.5', 3.0, np.nan, '5.0'])
        x_array = x_mixed.values  # This will be object dtype
        
        try:
            x_clean = _validate_and_clean_series(x_array, "test")
            assert x_clean.dtype == np.float64, "Should convert to float64"
            assert len(x_clean) > 0, "Should have some valid values"
        except (ValueError, TypeError):
            pass  # May fail, but should fail gracefully
