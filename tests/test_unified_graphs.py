"""
Tests for unified graph construction API.

Tests build_recurrence_graph, build_ordinal_partition_graph,
optimal_lag, and optimal_dim functions.
"""

import numpy as np
import pytest
import networkx as nx

from ts2net.viz import (
    build_recurrence_graph,
    build_ordinal_partition_graph,
    build_visibility_graph,
    optimal_lag,
    optimal_dim,
    TSGraph,
)


class TestBuildRecurrenceGraph:
    """Test build_recurrence_graph function."""
    
    def test_basic_recurrence_graph(self):
        """Test basic recurrence graph construction."""
        np.random.seed(42)
        x = np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50)
        tsgraph = build_recurrence_graph(x, embed_dim=3, delay=1, eps=0.2)
        
        assert isinstance(tsgraph, TSGraph)
        assert isinstance(tsgraph.graph, nx.Graph)
        assert tsgraph.meta["method"] == "recurrence"
        assert tsgraph.meta["embed_dim"] == 3
        assert tsgraph.meta["delay"] == 1
        assert tsgraph.graph.number_of_nodes() > 0
        assert tsgraph.graph.number_of_edges() > 0
    
    def test_recurrence_with_embedding(self):
        """Test recurrence graph with explicit embedding."""
        np.random.seed(42)
        x = np.random.randn(100)
        tsgraph = build_recurrence_graph(x, embed_dim=5, delay=2, eps=0.3)
        
        assert tsgraph.meta["embed_dim"] == 5
        assert tsgraph.meta["delay"] == 2
        # Check that nodes have state attributes
        for node in tsgraph.graph.nodes():
            assert "state" in tsgraph.graph.nodes[node]
            assert "t" in tsgraph.graph.nodes[node]
    
    def test_eps_mode_fraction_max(self):
        """Test epsilon mode: fraction_max."""
        x = np.sin(np.linspace(0, 2*np.pi, 30))
        tsgraph = build_recurrence_graph(x, embed_dim=3, eps=0.2, eps_mode="fraction_max")
        
        assert tsgraph.meta["eps_mode"] == "fraction_max"
        assert "eps_threshold" in tsgraph.meta
        assert tsgraph.meta["eps_threshold"] > 0
    
    def test_eps_mode_percentile(self):
        """Test epsilon mode: percentile."""
        x = np.sin(np.linspace(0, 2*np.pi, 30))
        tsgraph = build_recurrence_graph(x, embed_dim=3, eps=10.0, eps_mode="percentile")
        
        assert tsgraph.meta["eps_mode"] == "percentile"
        assert "eps_threshold" in tsgraph.meta
    
    def test_knn_mode_mutual(self):
        """Test kNN mode: mutual."""
        np.random.seed(42)
        x = np.random.randn(40)
        tsgraph = build_recurrence_graph(
            x, embed_dim=3, eps=0.1, knn=5, knn_mode="mutual"
        )
        
        assert tsgraph.meta["knn"] == 5
        assert tsgraph.meta["knn_mode"] == "mutual"
        # Mutual kNN should create symmetric edges
        assert tsgraph.graph.number_of_edges() > 0
    
    def test_knn_mode_directed(self):
        """Test kNN mode: directed."""
        np.random.seed(42)
        x = np.random.randn(40)
        tsgraph = build_recurrence_graph(
            x, embed_dim=3, eps=0.1, knn=5, knn_mode="directed"
        )
        
        assert isinstance(tsgraph.graph, nx.DiGraph)
        assert tsgraph.meta["knn_mode"] == "directed"
    
    def test_weighted_recurrence(self):
        """Test weighted recurrence graph."""
        x = np.sin(np.linspace(0, 2*np.pi, 30))
        tsgraph = build_recurrence_graph(
            x, embed_dim=3, eps=0.2, weighted=True, weight_mode="distance"
        )
        
        assert tsgraph.meta["weighted"] is True
        # Check that edges have weight attributes
        if tsgraph.graph.number_of_edges() > 0:
            edge = list(tsgraph.graph.edges(data=True))[0]
            assert "weight" in edge[2]
            assert "dist" in edge[2]
    
    def test_theiler_window(self):
        """Test Theiler window exclusion."""
        np.random.seed(42)
        x = np.random.randn(30)
        tsgraph1 = build_recurrence_graph(x, embed_dim=3, eps=0.2, theiler_window=0)
        tsgraph2 = build_recurrence_graph(x, embed_dim=3, eps=0.2, theiler_window=5)
        
        # Theiler window should reduce edges
        assert tsgraph2.graph.number_of_edges() <= tsgraph1.graph.number_of_edges()
    
    def test_exclude_diagonal(self):
        """Test diagonal exclusion."""
        np.random.seed(42)
        x = np.random.randn(20)
        tsgraph = build_recurrence_graph(x, embed_dim=3, eps=0.2, exclude_diagonal=True)
        
        # No self-loops should exist
        for node in tsgraph.graph.nodes():
            assert not tsgraph.graph.has_edge(node, node)
    
    def test_return_pos(self):
        """Test position return."""
        x = np.sin(np.linspace(0, 2*np.pi, 30))
        tsgraph = build_recurrence_graph(x, embed_dim=3, eps=0.2, return_pos=True)
        
        assert tsgraph.pos is not None
        assert len(tsgraph.pos) == tsgraph.graph.number_of_nodes()
        # Positions should be 2D arrays
        for pos in tsgraph.pos.values():
            assert len(pos) == 2
    
    def test_multivariate_input(self):
        """Test multivariate input (already embedded)."""
        np.random.seed(42)
        # Create 2D array (already embedded)
        embedded = np.random.randn(20, 3)
        tsgraph = build_recurrence_graph(embedded, embed_dim=3, eps=0.2)
        
        assert tsgraph.graph.number_of_nodes() == 20
    
    @pytest.mark.slow
    def test_large_n_incremental_mode(self):
        """Test that large n uses incremental distance computation."""
        # Large n should trigger incremental mode (n > 5000)
        np.random.seed(42)
        x = np.sin(np.linspace(0, 2*np.pi, 6000)) + 0.1 * np.random.randn(6000)
        
        # Should not raise memory error
        tsgraph = build_recurrence_graph(x, embed_dim=3, eps=0.1)
        
        # Invariants: graph builds, has nodes and edges
        assert tsgraph.graph.number_of_nodes() > 0
        assert tsgraph.graph.number_of_edges() > 0


class TestBuildOrdinalPartitionGraph:
    """Test build_ordinal_partition_graph function."""
    
    def test_basic_ordinal_partition(self):
        """Test basic ordinal partition graph construction."""
        np.random.seed(42)
        x = np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50)
        tsgraph = build_ordinal_partition_graph(x, embed_dim=4, delay=1)
        
        assert isinstance(tsgraph, TSGraph)
        assert isinstance(tsgraph.graph, nx.DiGraph)
        assert tsgraph.meta["method"] == "ordinal_partition"
        assert tsgraph.meta["embed_dim"] == 4
        assert tsgraph.graph.number_of_nodes() > 0
        assert tsgraph.graph.number_of_edges() > 0
    
    def test_pattern_attributes(self):
        """Test that nodes have pattern attributes."""
        np.random.seed(42)
        x = np.random.randn(30)
        tsgraph = build_ordinal_partition_graph(x, embed_dim=3, delay=1)
        
        for node in tsgraph.graph.nodes():
            assert "pattern" in tsgraph.graph.nodes[node]
            assert "count" in tsgraph.graph.nodes[node]
            # Pattern should be a tuple
            assert isinstance(tsgraph.graph.nodes[node]["pattern"], tuple)
    
    def test_weighted_transitions(self):
        """Test weighted transition edges."""
        x = np.sin(np.linspace(0, 2*np.pi, 40))
        tsgraph = build_ordinal_partition_graph(x, embed_dim=4, weighted=True)
        
        assert tsgraph.meta["weighted"] is True
        # Check that edges have weights
        if tsgraph.graph.number_of_edges() > 0:
            edge = list(tsgraph.graph.edges(data=True))[0]
            assert "weight" in edge[2]
            assert edge[2]["weight"] > 0
    
    def test_include_self_loops(self):
        """Test self-loop inclusion."""
        x = np.ones(30)  # Constant series should have self-loops
        tsgraph = build_ordinal_partition_graph(x, embed_dim=3, include_self_loops=True)
        
        # Constant series should have at least one pattern
        assert tsgraph.graph.number_of_nodes() > 0
    
    def test_tie_break_stable(self):
        """Test stable tie breaking."""
        x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])  # Ties present
        tsgraph = build_ordinal_partition_graph(x, embed_dim=3, tie_break="stable")
        
        assert tsgraph.meta["tie_break"] == "stable"
        assert tsgraph.graph.number_of_nodes() > 0
    
    def test_tie_break_jitter(self):
        """Test jitter tie breaking."""
        x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])  # Ties present
        tsgraph = build_ordinal_partition_graph(x, embed_dim=3, tie_break="jitter")
        
        assert tsgraph.meta["tie_break"] == "jitter"
        assert tsgraph.graph.number_of_nodes() > 0
    
    def test_undirected_ordinal(self):
        """Test undirected ordinal partition graph."""
        np.random.seed(42)
        x = np.random.randn(30)
        tsgraph = build_ordinal_partition_graph(x, embed_dim=4, directed=False)
        
        assert isinstance(tsgraph.graph, nx.Graph)
        assert tsgraph.meta["directed"] is False
    
    def test_return_pos_ordinal(self):
        """Test position return for ordinal graph."""
        np.random.seed(42)
        x = np.random.randn(30)
        tsgraph = build_ordinal_partition_graph(x, embed_dim=4, return_pos=True)
        
        assert tsgraph.pos is not None
        assert len(tsgraph.pos) == tsgraph.graph.number_of_nodes()
    
    def test_metadata_completeness(self):
        """Test that metadata is complete."""
        np.random.seed(42)
        x = np.random.randn(40)
        tsgraph = build_ordinal_partition_graph(x, embed_dim=5, delay=2)
        
        assert "n_original" in tsgraph.meta
        assert "n_windows" in tsgraph.meta
        assert "n_patterns" in tsgraph.meta
        assert "n_edges" in tsgraph.meta
        assert tsgraph.meta["n_patterns"] == tsgraph.graph.number_of_nodes()
        assert tsgraph.meta["n_edges"] == tsgraph.graph.number_of_edges()


class TestOptimalLag:
    """Test optimal_lag function."""
    
    def test_optimal_lag_sine(self):
        """Test optimal lag on sine wave."""
        x = np.sin(np.linspace(0, 4*np.pi, 200))
        tau = optimal_lag(x)
        
        # Invariant: tau should be a positive integer within bounds
        assert np.issubdtype(type(tau), np.integer) or isinstance(tau, int)
        assert tau > 0
        assert tau <= 50  # Should respect max_lag
    
    def test_optimal_lag_random(self):
        """Test optimal lag on random series."""
        np.random.seed(42)
        x = np.random.randn(100)
        tau = optimal_lag(x, max_lag=30)
        
        # Invariant: tau should be a positive integer within bounds
        assert np.issubdtype(type(tau), np.integer) or isinstance(tau, int)
        assert 1 <= tau <= 30
    
    def test_optimal_lag_short_series(self):
        """Test optimal lag on short series."""
        np.random.seed(42)
        x = np.random.randn(20)
        tau = optimal_lag(x, max_lag=10)
        
        # Invariant: tau should be a positive integer (numpy int or Python int)
        assert np.issubdtype(type(tau), np.integer) or isinstance(tau, int)
        assert tau > 0


class TestOptimalDim:
    """Test optimal_dim function."""
    
    def test_optimal_dim_sine(self):
        """Test optimal dimension on sine wave."""
        x = np.sin(np.linspace(0, 4*np.pi, 100))
        tau = optimal_lag(x)
        d = optimal_dim(x, delay=tau, dim_range=(3, 6))
        
        # Invariant: d should be an integer within range
        assert np.issubdtype(type(d), np.integer) or isinstance(d, int)
        assert 3 <= d <= 6
    
    def test_optimal_dim_random(self):
        """Test optimal dimension on random series."""
        np.random.seed(42)
        x = np.random.randn(80)
        d = optimal_dim(x, delay=1, dim_range=(2, 5))
        
        # Invariant: d should be an integer within range
        assert np.issubdtype(type(d), np.integer) or isinstance(d, int)
        assert 2 <= d <= 5
    
    def test_optimal_dim_custom_range(self):
        """Test optimal dimension with custom range."""
        x = np.sin(np.linspace(0, 2*np.pi, 60))
        d = optimal_dim(x, delay=1, dim_range=(4, 7))
        
        # Invariant: d should be an integer within range
        assert np.issubdtype(type(d), np.integer) or isinstance(d, int)
        assert 4 <= d <= 7


class TestUnifiedAPI:
    """Test that all graph types work with unified API."""
    
    def test_all_return_tsgraph(self):
        """Test that all builders return TSGraph."""
        x = np.sin(np.linspace(0, 2*np.pi, 50))
        
        tsgraph_viz = build_visibility_graph(x, kind="hvg")
        tsgraph_rec = build_recurrence_graph(x, embed_dim=3, eps=0.2)
        tsgraph_ord = build_ordinal_partition_graph(x, embed_dim=4)
        
        assert isinstance(tsgraph_viz, TSGraph)
        assert isinstance(tsgraph_rec, TSGraph)
        assert isinstance(tsgraph_ord, TSGraph)
    
    def test_all_have_meta(self):
        """Test that all graphs have metadata."""
        np.random.seed(42)
        x = np.random.randn(40)
        
        tsgraph_viz = build_visibility_graph(x, kind="hvg")
        tsgraph_rec = build_recurrence_graph(x, embed_dim=3, eps=0.2)
        tsgraph_ord = build_ordinal_partition_graph(x, embed_dim=4)
        
        assert "method" in tsgraph_viz.meta
        assert "method" in tsgraph_rec.meta
        assert "method" in tsgraph_ord.meta
    
    def test_error_handling_short_series(self):
        """Test error handling for too-short series."""
        x = np.array([1.0, 2.0])  # Too short for embedding
        
        with pytest.raises(ValueError):
            build_ordinal_partition_graph(x, embed_dim=4)
        
        # Recurrence should handle it gracefully
        tsgraph = build_recurrence_graph(x, embed_dim=1, eps=0.1)
        assert tsgraph.graph.number_of_nodes() > 0
