"""
Tests for multiscale graph analysis.

Tests coarse-graining and scale signature computation.
"""

import numpy as np
import pytest
from ts2net.multiscale import MultiscaleGraphs, coarse_grain


class TestCoarseGrain:
    """Test coarse-graining function."""
    
    def test_coarse_grain_mean(self):
        """Test mean aggregation."""
        x = np.arange(12.0)
        result = coarse_grain(x, scale=3, method="mean")
        expected = np.array([1.0, 4.0, 7.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_coarse_grain_median(self):
        """Test median aggregation."""
        x = np.arange(12.0)
        result = coarse_grain(x, scale=3, method="median")
        expected = np.array([1.0, 4.0, 7.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_coarse_grain_max(self):
        """Test max aggregation."""
        x = np.arange(12.0)
        result = coarse_grain(x, scale=3, method="max")
        expected = np.array([2.0, 5.0, 8.0, 11.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_coarse_grain_min(self):
        """Test min aggregation."""
        x = np.arange(12.0)
        result = coarse_grain(x, scale=3, method="min")
        expected = np.array([0.0, 3.0, 6.0, 9.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_coarse_grain_std(self):
        """Test std aggregation."""
        x = np.arange(12.0)
        result = coarse_grain(x, scale=3, method="std")
        # Standard deviation of [0,1,2] is sqrt(2/3) ≈ 0.816
        expected_std = np.std([0, 1, 2], ddof=1)
        assert abs(result[0] - expected_std) < 1e-10
    
    def test_coarse_grain_scale_1(self):
        """Test scale=1 returns original."""
        x = np.random.randn(100)
        result = coarse_grain(x, scale=1, method="mean")
        np.testing.assert_array_equal(result, x)
    
    def test_coarse_grain_invalid_scale(self):
        """Test invalid scale raises error."""
        x = np.arange(10)
        with pytest.raises(ValueError, match="scale must be positive"):
            coarse_grain(x, scale=0, method="mean")
        
        with pytest.raises(ValueError, match="scale.*must be less than"):
            coarse_grain(x, scale=20, method="mean")
    
    def test_coarse_grain_invalid_method(self):
        """Test invalid method raises error."""
        x = np.arange(10)
        with pytest.raises(ValueError, match="Unknown method"):
            coarse_grain(x, scale=2, method="invalid")


class TestMultiscaleGraphs:
    """Test MultiscaleGraphs class."""
    
    def test_basic_multiscale_hvg(self):
        """Test basic multiscale HVG analysis."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4])
        ms.fit(x)
        
        signature = ms.scale_signature()
        
        # Should have features at each scale
        assert 'n_nodes' in signature
        assert 'n_edges' in signature
        assert 'avg_degree' in signature
        
        # Should have 3 scales
        assert len(signature['n_nodes']) == 3
        assert len(signature['avg_degree']) == 3
        
        # All values should be finite
        assert np.all(np.isfinite(signature['n_nodes']))
        assert np.all(np.isfinite(signature['avg_degree']))
    
    def test_multiscale_default_scales(self):
        """Test default scales."""
        x = np.random.randn(500)
        
        ms = MultiscaleGraphs(method='hvg')
        ms.fit(x)
        
        signature = ms.scale_signature()
        
        # Default scales are [1, 2, 4, 8, 16]
        assert len(signature['n_nodes']) == 5
    
    def test_multiscale_nvg(self):
        """Test multiscale NVG analysis."""
        x = np.random.randn(300)
        
        ms = MultiscaleGraphs(method='nvg', scales=[1, 2, 4], output='stats')
        ms.fit(x)
        
        signature = ms.scale_signature()
        
        assert 'n_nodes' in signature
        assert 'n_edges' in signature
        assert len(signature['n_nodes']) == 3
    
    def test_multiscale_transition(self):
        """Test multiscale transition network."""
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='transition', scales=[1, 2, 4])
        ms.fit(x)
        
        signature = ms.scale_signature()
        
        assert 'n_nodes' in signature
        assert len(signature['n_nodes']) == 3
    
    def test_scale_signature_custom_features(self):
        """Test scale signature with custom features."""
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2])
        ms.fit(x)
        
        signature = ms.scale_signature(features=['n_nodes', 'density'])
        
        assert 'n_nodes' in signature
        assert 'density' in signature
        assert 'n_edges' not in signature
    
    def test_stats_per_scale(self):
        """Test getting full stats at each scale."""
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4])
        ms.fit(x)
        
        stats = ms.stats()
        
        assert 1 in stats
        assert 2 in stats
        assert 4 in stats
        
        # Each scale should have stats dict
        assert isinstance(stats[1], dict)
        assert 'n_nodes' in stats[1]
        assert 'n_edges' in stats[1]
    
    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2])
        signature = ms.fit_transform(x)
        
        assert 'n_nodes' in signature
        assert len(signature['n_nodes']) == 2
    
    def test_series_too_short(self):
        """Test error when series is too short."""
        x = np.random.randn(10)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4, 8, 16])
        
        with pytest.raises(ValueError, match="must be >="):
            ms.fit(x)
    
    def test_scale_too_large_skipped(self):
        """Test that scales that make series too short are skipped."""
        x = np.random.randn(50)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4, 8, 16, 32])
        ms.fit(x)
        
        stats = ms.stats()
        
        # Scale 32 might be skipped if it makes series too short
        # But smaller scales should work
        assert 1 in stats or stats[1] is None
        assert 2 in stats or stats[2] is None
    
    def test_different_coarse_methods(self):
        """Test different coarse-graining methods."""
        x = np.random.randn(200)
        
        for method in ['mean', 'median', 'max', 'min', 'std']:
            ms = MultiscaleGraphs(method='hvg', scales=[1, 2, 4], coarse_method=method)
            ms.fit(x)
            signature = ms.scale_signature()
            
            assert len(signature['n_nodes']) == 3
    
    def test_scale_signature_stability(self):
        """Test that scale signature is stable for similar series."""
        np.random.seed(42)
        x1 = np.random.randn(200)
        x2 = np.random.randn(200)  # Different but similar
        
        ms1 = MultiscaleGraphs(method='hvg', scales=[1, 2, 4])
        ms2 = MultiscaleGraphs(method='hvg', scales=[1, 2, 4])
        
        sig1 = ms1.fit_transform(x1)
        sig2 = ms2.fit_transform(x2)
        
        # Signatures should have same structure
        assert set(sig1.keys()) == set(sig2.keys())
        assert len(sig1['n_nodes']) == len(sig2['n_nodes'])
    
    def test_invalid_method(self):
        """Test error for invalid method."""
        x = np.random.randn(100)
        
        ms = MultiscaleGraphs(method='invalid', scales=[1, 2])
        
        with pytest.raises(ValueError, match="Unknown method"):
            ms.fit(x)
    
    def test_multiscale_with_weighted(self):
        """Test multiscale with weighted graphs."""
        x = np.random.randn(200)
        
        ms = MultiscaleGraphs(method='hvg', scales=[1, 2], weighted=True)
        ms.fit(x)
        
        signature = ms.scale_signature()
        
        # Should work with weighted graphs
        assert 'n_nodes' in signature
        assert len(signature['n_nodes']) == 2


