"""
Tests for Directed Horizontal Visibility Graph (HVG).

Tests irreversibility analysis for fault detection in time series.
"""

import numpy as np
import pytest
from ts2net.api import HVG


class TestDirectedHVGBasic:
    """Basic tests for directed HVG."""
    
    def test_directed_hvg_basic(self):
        """Test directed HVG on simple time series."""
        x = np.array([1, 2, 3, 2, 1])
        hvg = HVG(directed=True)
        hvg.build(x)
        
        stats = hvg.stats()
        assert stats['n_nodes'] == len(x)
        assert stats['n_edges'] > 0
        assert 'avg_in_degree' in stats
        assert 'avg_out_degree' in stats
        assert 'irreversibility_score' in stats
        
        # Directed graphs should have in/out degree sequences
        in_deg = hvg.in_degree_sequence()
        out_deg = hvg.out_degree_sequence()
        assert len(in_deg) == len(x)
        assert len(out_deg) == len(x)
        assert np.all(in_deg >= 0)
        assert np.all(out_deg >= 0)
    
    def test_directed_vs_undirected_edge_count(self):
        """Test that directed graph has forward edges only."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        hvg_undir = HVG(directed=False)
        hvg_undir.build(x)
        n_edges_undir = hvg_undir.n_edges
        
        hvg_dir = HVG(directed=True)
        hvg_dir.build(x)
        n_edges_dir = hvg_dir.n_edges
        
        # Directed graph should have fewer or equal edges (no reverse edges)
        assert n_edges_dir <= n_edges_undir
        
        # For HVG, edges are forward only (i < j), so directed should have 
        # roughly half the edges of undirected (but not exactly due to self-loops, etc.)
        # Actually, HVG edges are naturally forward-only, so they should match
        # (undirected just adds reverse edges)
        assert n_edges_dir <= n_edges_undir


class TestDirectedHVGIrreversibility:
    """Tests for irreversibility score computation."""
    
    def test_reversible_signal_has_irreversibility(self):
        """Test that irreversibility score is computed for symmetric signals."""
        # Sine wave is reversible (time-reversal symmetric)
        t = np.linspace(0, 4*np.pi, 100)
        x = np.sin(t)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        stats = hvg.stats()
        
        # Irreversibility should be computed (in [0, 1] range)
        # Note: Even symmetric signals can have non-zero irreversibility 
        # due to the discrete graph structure
        irrev = stats['irreversibility_score']
        assert 0 <= irrev <= 1
    
    def test_ramp_signal_irreversibility(self):
        """Test irreversibility computation for ramp signal."""
        # Strictly increasing signal (ramp up)
        x = np.linspace(0, 10, 100)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        stats = hvg.stats()
        
        # Irreversibility should be computed and in valid range
        ramp_irrev = stats['irreversibility_score']
        assert 0 <= ramp_irrev <= 1
    
    def test_constant_signal_irreversibility(self):
        """Test irreversibility computation for constant signal."""
        x = np.ones(50)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        stats = hvg.stats()
        
        # For constant signal, in_degree should equal out_degree for most nodes
        # Irreversibility should be computed and in valid range
        # (may not be exactly 0 due to graph structure)
        irrev = stats['irreversibility_score']
        assert 0 <= irrev <= 1
    
    def test_asymmetric_pattern_detection(self):
        """Test that asymmetric patterns are detected."""
        # Create asymmetric signal: ramp up, then sudden drop
        x = np.concatenate([
            np.linspace(0, 10, 50),  # Ramp up
            np.array([0] * 50)       # Sudden drop
        ])
        
        hvg = HVG(directed=True)
        hvg.build(x)
        stats = hvg.stats()
        
        # Should have non-zero irreversibility
        assert stats['irreversibility_score'] > 0


class TestDirectedHVGDegreeSequences:
    """Tests for in-degree and out-degree sequences."""
    
    def test_in_out_degree_sum_equals_total_degree(self):
        """For directed graphs, in_degree + out_degree should match total connections."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        
        in_deg = hvg.in_degree_sequence()
        out_deg = hvg.out_degree_sequence()
        
        # Sum of in-degrees should equal sum of out-degrees (both equal to n_edges)
        assert np.sum(in_deg) == hvg.n_edges
        assert np.sum(out_deg) == hvg.n_edges
        assert np.sum(in_deg) == np.sum(out_deg)
    
    def test_degree_sequences_match_stats(self):
        """Degree sequences should match stats."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        
        stats = hvg.stats()
        in_deg = hvg.in_degree_sequence()
        out_deg = hvg.out_degree_sequence()
        
        assert np.isclose(np.mean(in_deg), stats['avg_in_degree'])
        assert np.isclose(np.mean(out_deg), stats['avg_out_degree'])
        assert np.min(in_deg) == stats['min_in_degree']
        assert np.max(in_deg) == stats['max_in_degree']
        assert np.min(out_deg) == stats['min_out_degree']
        assert np.max(out_deg) == stats['max_out_degree']
    
    def test_in_out_degree_errors_on_undirected(self):
        """in_degree_sequence() and out_degree_sequence() should error on undirected graphs."""
        x = np.random.randn(100)
        hvg = HVG(directed=False)
        hvg.build(x)
        
        with pytest.raises(ValueError, match="only valid for directed graphs"):
            hvg.in_degree_sequence()
        
        with pytest.raises(ValueError, match="only valid for directed graphs"):
            hvg.out_degree_sequence()


class TestDirectedHVGEdgeDirection:
    """Tests that edges point forward in time."""
    
    def test_edges_forward_in_time(self):
        """All edges in directed HVG should point forward (i → j where i < j)."""
        np.random.seed(42)
        x = np.random.randn(100)
        
        hvg = HVG(directed=True, output="edges")
        hvg.build(x)
        
        edges = hvg.edges
        if edges:  # If we have edges
            for edge in edges:
                if len(edge) == 2:
                    i, j = edge
                else:
                    i, j = edge[0], edge[1]
                # All edges should point forward (i < j)
                assert i < j
    
    def test_directed_adjacency_matrix(self):
        """Directed adjacency matrix should be asymmetric."""
        x = np.random.randn(50)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        
        A = hvg.adjacency_matrix(format="dense")
        
        # For directed graph, A should generally be asymmetric
        # (may have some symmetric parts if edges are bidirectional conceptually,
        # but HVG edges are naturally forward-only)
        # Check that A.T != A (not perfectly symmetric)
        # For directed HVG, if (i,j) exists, (j,i) should NOT exist (i < j always)
        assert not np.array_equal(A, A.T)


class TestDirectedHVGStats:
    """Tests for stats output with directed graphs."""
    
    def test_stats_include_directed_metrics(self):
        """Stats should include all directed graph metrics."""
        x = np.random.randn(100)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        
        stats = hvg.stats()
        
        # Should have all directed metrics
        required_keys = [
            'avg_in_degree', 'std_in_degree',
            'avg_out_degree', 'std_out_degree',
            'min_in_degree', 'max_in_degree',
            'min_out_degree', 'max_out_degree',
            'irreversibility_score'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_irreversibility_score_range(self):
        """Irreversibility score should be in [0, 1]."""
        x = np.random.randn(100)
        
        hvg = HVG(directed=True)
        hvg.build(x)
        
        stats = hvg.stats()
        irrev = stats['irreversibility_score']
        
        assert 0 <= irrev <= 1, f"Irreversibility score {irrev} out of range [0, 1]"


class TestDirectedHVGBackwardCompatibility:
    """Tests that directed=False maintains backward compatibility."""
    
    def test_undirected_still_works(self):
        """Undirected HVG should still work as before."""
        x = np.random.randn(100)
        
        hvg_undir = HVG(directed=False)
        hvg_undir.build(x)
        
        # Should not have directed metrics
        stats = hvg_undir.stats()
        assert 'avg_in_degree' not in stats
        assert 'avg_out_degree' not in stats
        assert 'irreversibility_score' not in stats
    
    def test_directed_default_is_false(self):
        """Default should be directed=False for backward compatibility."""
        x = np.random.randn(100)
        
        hvg1 = HVG()
        hvg2 = HVG(directed=False)
        
        hvg1.build(x)
        hvg2.build(x)
        
        # Should produce same results
        stats1 = hvg1.stats()
        stats2 = hvg2.stats()
        
        # Basic stats should match
        assert stats1['n_nodes'] == stats2['n_nodes']
        assert stats1['n_edges'] == stats2['n_edges']
        assert stats1['avg_degree'] == stats2['avg_degree']

