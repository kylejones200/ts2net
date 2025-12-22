"""Simplified unit tests for visibility graph implementations."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core.visibility import HVG, NVG


class TestHVG:
    """Basic tests for Horizontal Visibility Graph (HVG)."""
    
    def test_hvg_basic(self):
        """Test HVG on a simple time series."""
        x = np.array([1, 2, 3, 2, 1])
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() > 0
        assert nx.is_connected(G)


class TestNVG:
    """Basic tests for Natural Visibility Graph (NVG)."""
    
    def test_nvg_basic(self):
        """Test NVG on a simple time series."""
        x = np.array([1, 2, 3, 2, 1])
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() > 0
        assert nx.is_connected(G)


class TestVisibilityGraphCommon:
    """Common basic tests for both HVG and NVG."""
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_empty_series(self, graph_class):
        """Test with empty input series."""
        x = np.array([])
        G, A = graph_class().fit_transform(x)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


class TestHVGAlgorithm:
    """Test HVG O(n) stack algorithm correctness."""
    
    def test_hvg_stack_algorithm_identical(self):
        """Verify O(n) stack algorithm produces identical edges to naive approach for small series."""
        # Use small series where we can verify correctness
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
        
        # Build with numba (O(n) stack algorithm)
        from ts2net.core.visibility.hvg import _hvg_edges_numba
        try:
            from numba import njit
            HAS_NUMBA = True
        except ImportError:
            HAS_NUMBA = False
        
        if HAS_NUMBA:
            rows_fast, cols_fast, weights_fast = _hvg_edges_numba(x, weighted=False, limit=-1)
            edges_fast = set(zip(rows_fast, cols_fast))
            
            # Build with naive approach (for verification)
            edges_naive = set()
            n = len(x)
            for i in range(n):
                for j in range(i + 1, n):
                    # Check horizontal visibility
                    visible = True
                    for k in range(i + 1, j):
                        if x[k] > min(x[i], x[j]):
                            visible = False
                            break
                    if visible:
                        edges_naive.add((i, j))
            
            # Should match (allowing for undirected graph symmetry)
            # Fast algorithm may have different edge direction, so check both
            edges_fast_symmetric = edges_fast | {(j, i) for i, j in edges_fast}
            assert edges_fast_symmetric == edges_naive, \
                f"Stack algorithm edges don't match naive: {edges_fast_symmetric} vs {edges_naive}"
    
    def test_hvg_linear_scaling(self):
        """Verify HVG scales linearly (approximately 2n edges)."""
        np.random.seed(42)
        for n in [100, 1000, 10000]:
            x = np.random.randn(n)
            hvg = HVG()
            G, A = hvg.fit_transform(x)
            # HVG should have approximately 2n edges
            assert G.number_of_edges() < 3 * n, \
                f"HVG has too many edges ({G.number_of_edges()}) for n={n}, expected ~2n"
            assert G.number_of_edges() > n, \
                f"HVG has too few edges ({G.number_of_edges()}) for n={n}"
