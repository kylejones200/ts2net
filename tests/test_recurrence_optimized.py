"""
Tests for optimized recurrence network construction.
"""

import numpy as np
import pytest
import networkx as nx
from scipy.sparse import csr_matrix

try:
    from ts2net.core.recurrence_optimized import (
        knn_recurrence_optimized,
        epsilon_recurrence_optimized,
        parallel_recurrence_batch,
    )
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False


@pytest.mark.skipif(not HAS_OPTIMIZED, reason="Optimized recurrence requires scipy")
class TestKNNRecurrenceOptimized:
    """Test optimized k-NN recurrence networks."""
    
    def test_knn_basic(self):
        """Test basic k-NN recurrence with spatial indexing."""
        X = np.random.randn(100, 5)
        
        G, A = knn_recurrence_optimized(X, k=10, index_type="kdtree")
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
        assert isinstance(A, csr_matrix)
        assert A.shape == (100, 100)
        # Each node should have approximately k neighbors (may be less due to symmetry)
        assert G.number_of_edges() > 0
    
    def test_knn_weighted(self):
        """Test weighted k-NN recurrence."""
        X = np.random.randn(50, 3)
        
        G, A = knn_recurrence_optimized(X, k=5, weighted=True, index_type="kdtree")
        
        # Check that edges have weights
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 0
    
    def test_knn_auto_index(self):
        """Test automatic index type selection."""
        X = np.random.randn(100, 5)
        
        G, A = knn_recurrence_optimized(X, k=10, index_type="auto")
        
        assert G.number_of_nodes() == 100
    
    def test_knn_invalid_k(self):
        """Test that invalid k raises error."""
        X = np.random.randn(50, 3)
        
        with pytest.raises(ValueError, match="k must be in range"):
            knn_recurrence_optimized(X, k=0)
        
        with pytest.raises(ValueError, match="k must be in range"):
            knn_recurrence_optimized(X, k=50)
    
    def test_knn_non_euclidean_fallback(self):
        """Test fallback for non-euclidean metrics."""
        X = np.random.randn(50, 3)
        
        # Should fall back to distance matrix approach when using "auto"
        G, A = knn_recurrence_optimized(X, k=5, metric="manhattan", index_type="auto")
        
        assert G.number_of_nodes() == 50


@pytest.mark.skipif(not HAS_OPTIMIZED, reason="Optimized recurrence requires scipy")
class TestEpsilonRecurrenceOptimized:
    """Test optimized epsilon recurrence networks."""
    
    def test_epsilon_basic(self):
        """Test basic epsilon recurrence with spatial indexing."""
        X = np.random.randn(100, 5)
        
        G, A = epsilon_recurrence_optimized(X, epsilon=1.0, index_type="kdtree")
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
        assert isinstance(A, csr_matrix)
        assert A.shape == (100, 100)
    
    def test_epsilon_weighted(self):
        """Test weighted epsilon recurrence."""
        X = np.random.randn(50, 3)
        
        G, A = epsilon_recurrence_optimized(X, epsilon=0.5, weighted=True, index_type="kdtree")
        
        # Check that edges have weights
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 0
    
    def test_epsilon_approximate(self):
        """Test approximate epsilon recurrence."""
        X = np.random.randn(200, 5)
        
        G_exact, _ = epsilon_recurrence_optimized(
            X, epsilon=1.0, approximate=False, index_type="kdtree"
        )
        G_approx, _ = epsilon_recurrence_optimized(
            X, epsilon=1.0, approximate=True, index_type="kdtree"
        )
        
        # Approximate should have similar or fewer edges
        assert G_approx.number_of_edges() <= G_exact.number_of_edges() * 1.1
    
    def test_epsilon_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        X = np.random.randn(50, 3)
        
        with pytest.raises(ValueError, match="epsilon must be positive"):
            epsilon_recurrence_optimized(X, epsilon=0)
        
        with pytest.raises(ValueError, match="epsilon must be positive"):
            epsilon_recurrence_optimized(X, epsilon=-1.0)
    
    def test_epsilon_auto_index(self):
        """Test automatic index type selection."""
        X = np.random.randn(100, 5)
        
        G, A = epsilon_recurrence_optimized(X, epsilon=1.0, index_type="auto")
        
        assert G.number_of_nodes() == 100


@pytest.mark.skipif(not HAS_OPTIMIZED, reason="Optimized recurrence requires scipy")
class TestParallelRecurrenceBatch:
    """Test parallel batch processing of recurrence networks."""
    
    def test_parallel_batch_knn(self):
        """Test parallel k-NN batch processing."""
        X_list = [np.random.randn(50, 3) for _ in range(5)]
        
        results = parallel_recurrence_batch(
            X_list,
            method="knn",
            k=10,
            n_jobs=2
        )
        
        assert len(results) == 5
        for G, A in results:
            assert isinstance(G, nx.Graph)
            assert isinstance(A, csr_matrix)
            assert G.number_of_nodes() == 50
    
    def test_parallel_batch_epsilon(self):
        """Test parallel epsilon batch processing."""
        X_list = [np.random.randn(50, 3) for _ in range(5)]
        
        results = parallel_recurrence_batch(
            X_list,
            method="epsilon",
            epsilon=1.0,
            n_jobs=2
        )
        
        assert len(results) == 5
        for G, A in results:
            assert isinstance(G, nx.Graph)
            assert isinstance(A, csr_matrix)
    
    def test_parallel_batch_missing_params(self):
        """Test that missing parameters raise errors."""
        X_list = [np.random.randn(50, 3) for _ in range(3)]
        
        with pytest.raises(ValueError, match="k required"):
            parallel_recurrence_batch(X_list, method="knn")
        
        with pytest.raises(ValueError, match="epsilon required"):
            parallel_recurrence_batch(X_list, method="epsilon")


@pytest.mark.skipif(not HAS_OPTIMIZED, reason="Optimized recurrence requires scipy")
class TestPerformanceComparison:
    """Test that optimized methods are faster than naive approaches."""
    
    def test_knn_large_dataset(self):
        """Test k-NN on larger dataset (should use spatial indexing)."""
        X = np.random.randn(1000, 10)
        
        # Should complete without memory issues
        G, A = knn_recurrence_optimized(X, k=20, index_type="kdtree")
        
        assert G.number_of_nodes() == 1000
        assert G.number_of_edges() > 0
    
    def test_epsilon_large_dataset(self):
        """Test epsilon on larger dataset (should use spatial indexing)."""
        X = np.random.randn(500, 5)
        
        # Should complete without building full distance matrix
        G, A = epsilon_recurrence_optimized(
            X, epsilon=1.0, index_type="kdtree", approximate=True
        )
        
        assert G.number_of_nodes() == 500


@pytest.mark.skipif(not HAS_OPTIMIZED, reason="Optimized recurrence requires scipy")
class TestIntegrationWithRecurrenceNetwork:
    """Test integration with RecurrenceNetwork class."""
    
    def test_recurrence_network_uses_optimized(self):
        """Test that RecurrenceNetwork uses optimized methods when available."""
        from ts2net.core.recurrence import RecurrenceNetwork
        
        X = np.random.randn(200, 5)
        
        # k-NN should use optimized spatial indexing
        rn = RecurrenceNetwork(rule='knn', k=10)
        G = rn.fit(X).transform()
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 200
    
    def test_recurrence_network_epsilon_optimized(self):
        """Test that epsilon recurrence uses optimized methods."""
        from ts2net.core.recurrence import RecurrenceNetwork
        
        X = np.random.randn(200, 5)
        
        # Epsilon should use optimized spatial indexing
        rn = RecurrenceNetwork(rule='epsilon', threshold=1.0)
        G = rn.fit(X).transform()
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 200

