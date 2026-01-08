"""
Tests for feature-wise network comparisons.
"""

import numpy as np
import pytest
import pandas as pd

try:
    from ts2net.multivariate.feature_comparison import (
        compute_network_features,
        compare_network_features,
        cluster_series_by_features,
    )
    HAS_FEATURE_COMPARISON = True
except ImportError:
    HAS_FEATURE_COMPARISON = False


@pytest.mark.skipif(not HAS_FEATURE_COMPARISON, reason="Feature comparison requires pandas")
class TestComputeNetworkFeatures:
    """Test network feature computation for multiple series."""
    
    def test_hvg_features(self):
        """Test HVG feature computation."""
        X = [np.random.randn(100) for _ in range(3)]
        df = compute_network_features(X, method="hvg")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "n_nodes" in df.columns
        assert "n_edges" in df.columns
        assert "density" in df.columns
    
    def test_nvg_features(self):
        """Test NVG feature computation."""
        X = [np.random.randn(100) for _ in range(3)]
        df = compute_network_features(X, method="nvg")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "n_nodes" in df.columns
    
    def test_recurrence_features(self):
        """Test recurrence network feature computation."""
        X = [np.random.randn(100) for _ in range(3)]
        df = compute_network_features(X, method="recurrence", k=10)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "n_nodes" in df.columns
    
    def test_custom_series_names(self):
        """Test with custom series names."""
        X = [np.random.randn(50) for _ in range(2)]
        names = ["Series_A", "Series_B"]
        df = compute_network_features(X, method="hvg", series_names=names)
        
        assert list(df.index) == names
    
    def test_2d_array_input(self):
        """Test with 2D array input."""
        X = np.random.randn(3, 100)
        df = compute_network_features(X, method="hvg")
        
        assert len(df) == 3
    
    def test_with_kwargs(self):
        """Test with additional network builder kwargs."""
        X = [np.random.randn(100) for _ in range(2)]
        df = compute_network_features(X, method="hvg", weighted=True, directed=True)
        
        assert len(df) == 2


@pytest.mark.skipif(not HAS_FEATURE_COMPARISON, reason="Feature comparison requires pandas")
class TestCompareNetworkFeatures:
    """Test network feature comparison."""
    
    def test_basic_comparison(self):
        """Test basic feature comparison."""
        X = [np.random.randn(100) for _ in range(5)]
        df = compute_network_features(X, method="hvg")
        comparison = compare_network_features(df)
        
        assert isinstance(comparison, dict)
        assert "density" in comparison
        assert "mean" in comparison["density"]
        assert "std" in comparison["density"]
        assert "cv" in comparison["density"]
    
    def test_single_metric_comparison(self):
        """Test comparison for single metric."""
        X = [np.random.randn(100) for _ in range(5)]
        df = compute_network_features(X, method="hvg")
        comparison = compare_network_features(df, metric="density")
        
        assert "density" in comparison
        assert "mean" in comparison["density"]
    
    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        X = [np.random.randn(100) for _ in range(3)]
        df = compute_network_features(X, method="hvg")
        comparison = compare_network_features(df)
        
        # Should have similarity matrix if multiple metrics
        if "similarity_matrix" in comparison:
            assert isinstance(comparison["similarity_matrix"], dict)


@pytest.mark.skipif(not HAS_FEATURE_COMPARISON, reason="Feature comparison requires pandas")
class TestClusterSeriesByFeatures:
    """Test series clustering by network features."""
    
    def test_kmeans_clustering(self):
        """Test k-means clustering."""
        X = [np.random.randn(100) for _ in range(5)]
        df = compute_network_features(X, method="hvg")
        clusters = cluster_series_by_features(df, n_clusters=2, method="kmeans")
        
        assert isinstance(clusters, dict)
        assert len(clusters) == 5
        assert all(isinstance(v, int) for v in clusters.values())
        assert len(set(clusters.values())) <= 2
    
    def test_hierarchical_clustering(self):
        """Test hierarchical clustering."""
        X = [np.random.randn(100) for _ in range(5)]
        df = compute_network_features(X, method="hvg")
        clusters = cluster_series_by_features(df, n_clusters=2, method="hierarchical")
        
        assert isinstance(clusters, dict)
        assert len(clusters) == 5
        assert len(set(clusters.values())) <= 2
    
    def test_auto_clusters(self):
        """Test automatic cluster number determination."""
        X = [np.random.randn(100) for _ in range(5)]
        df = compute_network_features(X, method="hvg")
        clusters = cluster_series_by_features(df, method="kmeans")
        
        assert isinstance(clusters, dict)
        assert len(clusters) == 5


@pytest.mark.skipif(not HAS_FEATURE_COMPARISON, reason="Feature comparison requires pandas")
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series_list(self):
        """Test with empty series list."""
        X = []
        df = compute_network_features(X, method="hvg")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_single_series(self):
        """Test with single series."""
        X = [np.random.randn(100)]
        df = compute_network_features(X, method="hvg")
        
        assert len(df) == 1
    
    def test_different_length_series(self):
        """Test with series of different lengths."""
        X = [np.random.randn(100), np.random.randn(200), np.random.randn(50)]
        df = compute_network_features(X, method="hvg")
        
        assert len(df) == 3
    
    def test_invalid_method(self):
        """Test with invalid method."""
        X = [np.random.randn(100)]
        with pytest.raises(ValueError):
            compute_network_features(X, method="invalid")
    
    def test_mismatched_names(self):
        """Test with mismatched series names."""
        X = [np.random.randn(100) for _ in range(2)]
        names = ["A", "B", "C"]  # Too many names
        with pytest.raises(ValueError, match="must match"):
            compute_network_features(X, method="hvg", series_names=names)

