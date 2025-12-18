"""Integration tests for core module with distance metrics."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core import batch_transform
from ts2net.distances.core import tsdist_cor, tsdist_dtw, dist_matrix_normalize

class TestCoreWithDistances:
    """Test integration between core module and distance metrics."""
    
    @pytest.fixture
    def sample_time_series(self):
        """Generate sample time series for testing."""
        np.random.seed(42)
        return [
            np.random.randn(20) for _ in range(5)  # 5 time series, 20 points each
        ]
    
    def test_batch_transform_with_correlation(self, sample_time_series):
        """Test batch transform with correlation distance."""
        # Define a builder that uses correlation distance
        def corr_builder(series, **kwargs):
            """Build a graph using correlation distance."""
            from scipy.spatial.distance import squareform
            
            # Compute correlation distance
            D = 1 - np.corrcoef(series.reshape(1, -1))[0, 0]
            
            # Create a simple graph (just for testing)
            G = nx.Graph()
            G.add_node(0)
            G.graph['distance_matrix'] = np.array([[D]])
            return G
        
        # Register the builder
        import ts2net.core
        original_builders = ts2net.core._BUILDERS
        ts2net.core._BUILDERS = {"corr_test": corr_builder}
        
        try:
            # Test with a single time series
            results = batch_transform(
                [sample_time_series[0]], 
                builder="corr_test"
            )
            
            assert len(results) == 1
            assert isinstance(results[0], nx.Graph)
            assert 'distance_matrix' in results[0].graph
            
        finally:
            # Restore original builders
            ts2net.core._BUILDERS = original_builders
    
    def test_batch_transform_with_dtw(self, sample_time_series):
        """Test batch transform with DTW distance."""
        # Define a builder that uses DTW distance
        def dtw_builder(series, **kwargs):
            """Build a graph using DTW distance."""
            from ts2net.distances.core import tsdist_dtw
            
            # Compute DTW distance (just to self for testing)
            D = tsdist_dtw(series.reshape(1, -1))
            
            # Create a simple graph
            G = nx.Graph()
            G.add_node(0)
            G.graph['distance_matrix'] = D
            return G
        
        # Register the builder
        import ts2net.core
        original_builders = ts2net.core._BUILDERS
        ts2net.core._BUILDERS = {"dtw_test": dtw_builder}
        
        try:
            # Test with a single time series
            results = batch_transform(
                [sample_time_series[0]], 
                builder="dtw_test"
            )
            
            assert len(results) == 1
            assert isinstance(results[0], nx.Graph)
            assert 'distance_matrix' in results[0].graph
            
        finally:
            # Restore original builders
            ts2net.core._BUILDERS = original_builders
    
    def test_distance_matrix_normalization(self):
        """Test integration of distance matrix normalization."""
        # Create a sample distance matrix
        D = np.array([
            [0, 2, 3],
            [2, 0, 4],
            [3, 4, 0]
        ])
        
        # Test min-max normalization
        D_norm = dist_matrix_normalize(D, kind='minmax')
        assert np.min(D_norm) == 0.0
        assert np.max(D_norm) == 1.0
        
        # Test z-score normalization
        D_zscore = dist_matrix_normalize(D, kind='zscore')
        assert np.isclose(np.mean(D_zscore[np.triu_indices(3, k=1)]), 0.0, atol=1e-8)
        
        # Test that diagonal remains zero
        assert np.all(np.diag(D_norm) == 0.0)
        assert np.all(np.diag(D_zscore) == 0.0)
