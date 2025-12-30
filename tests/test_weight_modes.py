"""
Comprehensive tests for enhanced weighted visibility graphs.

Tests all weight modes: absdiff, time_gap, slope, min_clearance
and verifies weight statistics are included in stats().
"""

import numpy as np
import pytest
from ts2net import HVG, NVG


class TestWeightModes:
    """Test all weight modes for HVG and NVG."""
    
    def test_hvg_absdiff(self):
        """Test HVG with absdiff weight mode."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="absdiff")
        hvg.build(x)
        
        assert hvg.weighted
        assert hvg.weight_mode == "absdiff"
        
        # Check weights exist
        edges = hvg.edges
        assert all(len(e) == 3 for e in edges)  # All edges have weights
        
        # Check weight statistics
        stats = hvg.stats()
        assert 'min_weight' in stats
        assert 'max_weight' in stats
        assert 'mean_weight' in stats
        assert 'std_weight' in stats
        assert stats['min_weight'] >= 0  # Absolute differences are non-negative
    
    def test_hvg_time_gap(self):
        """Test HVG with time_gap weight mode."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="time_gap")
        hvg.build(x)
        
        assert hvg.weighted
        assert hvg.weight_mode == "time_gap"
        
        # Check weights exist
        edges = hvg.edges
        assert all(len(e) == 3 for e in edges)
        
        # Time gaps should be positive integers
        weights = [e[2] for e in edges]
        assert all(w > 0 for w in weights)
        assert all(w == int(w) for w in weights)  # Time gaps are integers
        
        # Check weight statistics
        stats = hvg.stats()
        assert 'min_weight' in stats
        assert stats['min_weight'] >= 1  # Minimum time gap is at least 1
    
    def test_hvg_slope(self):
        """Test HVG with slope weight mode."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="slope")
        hvg.build(x)
        
        assert hvg.weighted
        assert hvg.weight_mode == "slope"
        
        # Check weights exist
        edges = hvg.edges
        assert all(len(e) == 3 for e in edges)
        
        # Slopes can be positive or negative
        weights = [e[2] for e in edges]
        assert all(np.isfinite(w) for w in weights)
        
        # Check weight statistics
        stats = hvg.stats()
        assert 'min_weight' in stats
        assert 'max_weight' in stats
    
    def test_hvg_min_clearance(self):
        """Test HVG with min_clearance weight mode."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="min_clearance")
        hvg.build(x)
        
        assert hvg.weighted
        assert hvg.weight_mode == "min_clearance"
        
        # Check weights exist
        edges = hvg.edges
        assert all(len(e) == 3 for e in edges)
        
        # Min clearance can be positive, negative, or inf
        weights = [e[2] for e in edges]
        # Some weights might be inf (no intermediate points)
        finite_weights = [w for w in weights if np.isfinite(w)]
        if finite_weights:
            # Finite clearances can be negative (if intermediate point is above baseline)
            assert all(isinstance(w, (int, float)) for w in finite_weights)
        
        # Check weight statistics
        stats = hvg.stats()
        assert 'min_weight' in stats
        assert 'max_weight' in stats
    
    def test_hvg_weighted_bool_defaults_to_absdiff(self):
        """Test that weighted=True defaults to absdiff mode."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted=True)
        hvg.build(x)
        
        assert hvg.weighted
        assert hvg.weight_mode == "absdiff"
    
    def test_hvg_unweighted(self):
        """Test unweighted HVG."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted=False)
        hvg.build(x)
        
        assert not hvg.weighted
        assert hvg.weight_mode is None
        
        # Edges should not have weights
        edges = hvg.edges
        assert all(len(e) == 2 for e in edges)  # No weights
        
        # Weight statistics should not be present
        stats = hvg.stats()
        assert 'min_weight' not in stats
        assert 'max_weight' not in stats
    
    def test_nvg_all_weight_modes(self):
        """Test NVG with all weight modes."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        
        for mode in ["absdiff", "time_gap", "slope", "min_clearance"]:
            nvg = NVG(weighted=mode)
            nvg.build(x)
            
            assert nvg.weighted
            assert nvg.weight_mode == mode
            
            # Check weights exist
            edges = nvg.edges
            assert all(len(e) == 3 for e in edges)
            
            # Check weight statistics
            stats = nvg.stats()
            assert 'min_weight' in stats
            assert 'max_weight' in stats
            assert 'mean_weight' in stats
            assert 'std_weight' in stats
    
    def test_weight_mode_consistency(self):
        """Test that different weight modes produce same graph structure."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        
        hvg1 = HVG(weighted="absdiff")
        hvg2 = HVG(weighted="time_gap")
        hvg3 = HVG(weighted="slope")
        hvg4 = HVG(weighted="min_clearance")
        
        hvg1.build(x)
        hvg2.build(x)
        hvg3.build(x)
        hvg4.build(x)
        
        # All should have same number of edges (same graph structure)
        assert hvg1.n_edges == hvg2.n_edges
        assert hvg2.n_edges == hvg3.n_edges
        assert hvg3.n_edges == hvg4.n_edges
        
        # But different weights
        weights1 = [e[2] for e in hvg1.edges]
        weights2 = [e[2] for e in hvg2.edges]
        weights3 = [e[2] for e in hvg3.edges]
        weights4 = [e[2] for e in hvg4.edges]
        
        # Weights should be different (unless coincidentally equal)
        # At least check they're computed
        assert len(weights1) == len(weights2) == len(weights3) == len(weights4)
    
    def test_weight_statistics_in_stats(self):
        """Test that weight statistics are included in stats() output."""
        x = np.random.randn(100)
        
        for mode in ["absdiff", "time_gap", "slope", "min_clearance"]:
            hvg = HVG(weighted=mode)
            hvg.build(x)
            
            stats = hvg.stats()
            
            # Check all weight statistics are present
            assert 'min_weight' in stats
            assert 'max_weight' in stats
            assert 'mean_weight' in stats
            assert 'std_weight' in stats
            
            # Check statistics are reasonable
            assert stats['min_weight'] <= stats['mean_weight'] <= stats['max_weight']
            assert stats['std_weight'] >= 0
    
    def test_invalid_weight_mode(self):
        """Test that invalid weight mode raises error."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="invalid_mode")
        # Error should be raised when building, not at initialization
        with pytest.raises(ValueError, match="Unknown weight mode|Invalid"):
            hvg.build(x)
    
    def test_weight_mode_with_directed(self):
        """Test weight modes work with directed graphs."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        
        for mode in ["absdiff", "time_gap", "slope", "min_clearance"]:
            hvg = HVG(weighted=mode, directed=True)
            hvg.build(x)
            
            assert hvg.weighted
            assert hvg.weight_mode == mode
            assert hvg.directed
            
            # Check weights exist
            edges = hvg.edges
            assert all(len(e) == 3 for e in edges)
            
            # Check weight statistics
            stats = hvg.stats()
            assert 'min_weight' in stats
            assert 'max_weight' in stats

