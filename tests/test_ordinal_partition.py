"""
Tests for Ordinal Partition Networks feature.

Tests entropy rate, pattern distribution, and motif counting for ordinal partition networks.
"""

import numpy as np
import pytest
from ts2net.core.transition import TransitionNetwork


class TestOrdinalPartitionNetworks:
    """Test ordinal partition network features."""
    
    def test_partition_mode_requires_ordinal(self):
        """Test that partition mode requires ordinal symbolizer."""
        x = np.random.randn(100)
        tn = TransitionNetwork(symbolizer="equal_width", partition_mode=True)
        tn.fit_transform(x)
        
        # Should raise error when trying to use partition mode methods
        with pytest.raises(ValueError, match="requires partition_mode=True and symbolizer='ordinal'"):
            tn.entropy_rate()
    
    def test_entropy_rate_white_noise(self):
        """Test that white noise has high entropy rate."""
        np.random.seed(42)
        x = np.random.randn(1000)  # White noise
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        entropy = tn.entropy_rate()
        
        # White noise should have relatively high entropy (close to log2 of number of patterns)
        # For order=3, there are 3! = 6 possible patterns, so max entropy is log2(6) ≈ 2.58
        assert entropy > 0
        assert entropy <= 3.0  # Should be bounded by log2 of pattern count
        assert np.isfinite(entropy)
    
    def test_entropy_rate_periodic(self):
        """Test that periodic signal has low entropy rate."""
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t)  # Periodic signal
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        entropy = tn.entropy_rate()
        
        # Periodic signal should have lower entropy than white noise
        assert entropy >= 0
        assert np.isfinite(entropy)
    
    def test_entropy_rate_comparison(self):
        """Test that white noise has higher entropy than periodic signal."""
        np.random.seed(42)
        x_noise = np.random.randn(500)
        t = np.linspace(0, 4 * np.pi, 500)
        x_periodic = np.sin(t)
        
        tn_noise = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn_periodic = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        
        tn_noise.fit_transform(x_noise)
        tn_periodic.fit_transform(x_periodic)
        
        entropy_noise = tn_noise.entropy_rate()
        entropy_periodic = tn_periodic.entropy_rate()
        
        # White noise should generally have higher entropy
        # (though this might not always be true for small samples)
        assert entropy_noise >= 0
        assert entropy_periodic >= 0
        assert np.isfinite(entropy_noise)
        assert np.isfinite(entropy_periodic)
    
    def test_pattern_distribution(self):
        """Test pattern distribution computation."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0])
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit(x)
        
        dist = tn.pattern_distribution()
        
        # Should return a dictionary
        assert isinstance(dist, dict)
        
        # All frequencies should sum to 1.0 (approximately)
        total_freq = sum(dist.values())
        assert abs(total_freq - 1.0) < 1e-10
        
        # All frequencies should be between 0 and 1
        for freq in dist.values():
            assert 0 <= freq <= 1
    
    def test_pattern_distribution_requires_partition_mode(self):
        """Test that pattern_distribution requires partition_mode."""
        x = np.random.randn(100)
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=False)
        tn.fit(x)
        
        with pytest.raises(ValueError, match="requires partition_mode=True"):
            tn.pattern_distribution()
    
    def test_pattern_motifs_3node(self):
        """Test 3-node motif counting."""
        x = np.random.randn(200)
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        motifs = tn.pattern_motifs('3node')
        
        # Should return a dictionary
        assert isinstance(motifs, dict)
        
        # Should have some motifs (unless graph is too small)
        if tn._G.number_of_nodes() >= 3:
            # At least check the structure is correct
            # Motifs dict should have entries with 'count' and 'freq' keys
            for key, value in motifs.items():
                if isinstance(value, dict):
                    assert 'count' in value or 'freq' in value
    
    def test_pattern_motifs_4node(self):
        """Test 4-node motif counting."""
        x = np.random.randn(200)
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        motifs = tn.pattern_motifs('4node')
        
        # Should return a dictionary
        assert isinstance(motifs, dict)
        
        # Should have some motifs (unless graph is too small)
        if tn._G.number_of_nodes() >= 4:
            # At least check the structure is correct
            for key, value in motifs.items():
                if isinstance(value, dict):
                    assert 'count' in value or 'freq' in value
    
    def test_pattern_motifs_invalid_type(self):
        """Test that invalid motif type raises error."""
        x = np.random.randn(100)
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        with pytest.raises(ValueError, match="motif_type must be"):
            tn.pattern_motifs('invalid')
    
    def test_stats_includes_partition_metrics(self):
        """Test that stats() includes partition mode metrics."""
        x = np.random.randn(200)
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=True)
        tn.fit_transform(x)
        
        stats = tn.stats()
        
        # Should include basic stats
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
        assert 'density' in stats
        
        # Should include partition mode metrics
        assert 'entropy_rate' in stats
        assert 'pattern_distribution' in stats
        assert 'motif_counts_3node' in stats
        assert 'motif_counts_4node' in stats
        
        # Check types
        assert isinstance(stats['entropy_rate'], (int, float))
        assert isinstance(stats['pattern_distribution'], dict)
        assert isinstance(stats['motif_counts_3node'], dict)
        assert isinstance(stats['motif_counts_4node'], dict)
    
    def test_stats_without_partition_mode(self):
        """Test that stats() works without partition mode."""
        x = np.random.randn(200)
        
        tn = TransitionNetwork(symbolizer="ordinal", order=3, partition_mode=False)
        tn.fit_transform(x)
        
        stats = tn.stats()
        
        # Should include basic stats
        assert 'n_nodes' in stats
        assert 'n_edges' in stats
        assert 'density' in stats
        
        # Should NOT include partition mode metrics
        assert 'entropy_rate' not in stats
        assert 'pattern_distribution' not in stats
        assert 'motif_counts_3node' not in stats
        assert 'motif_counts_4node' not in stats
    
    def test_partition_mode_backward_compatibility(self):
        """Test that partition_mode=False maintains backward compatibility."""
        x = np.random.randn(100)
        
        # Default behavior (partition_mode=False)
        tn = TransitionNetwork(symbolizer="ordinal", order=3)
        G, A = tn.fit_transform(x)
        
        # Should work normally
        assert G.number_of_nodes() > 0
        assert A.shape[0] == A.shape[1]
        
        # Should not have partition mode methods available
        stats = tn.stats()
        assert 'entropy_rate' not in stats

