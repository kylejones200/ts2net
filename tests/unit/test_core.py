"""Unit tests for core module functionality."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core import (
    _vis_weights,
    _graph_from_edges_weighted,
    _ordinal_patterns,
    _adj_to_graph,
    _hvg_edges,
    triangle_count,
    wedge_count,
    motif_summary,
    _giant_component,
    small_world_summary,
    directed_3node_motifs,
    undirected_4node_motifs,
    graph_summary,
    batch_transform,
)

class TestCoreUtilities:
    """Test cases for core utility functions."""
    
    def test_vis_weights(self):
        """Test visibility weights calculation."""
        y = np.array([1, 2, 1, 3, 1])
        E = np.array([(0, 1), (1, 2), (2, 3), (3, 4)])
        
        # Test 'distance' mode
        w_dist = _vis_weights(y, E, mode='distance')
        assert len(w_dist) == len(E)
        assert np.allclose(w_dist, [1.0, 1.0, 1.0, 1.0])
        
        # Test 'product' mode
        w_prod = _vis_weights(y, E, mode='product')
        assert len(w_prod) == len(E)
        assert np.allclose(w_prod, [2.0, 2.0, 3.0, 3.0])
        
        # Test 'mean' mode
        w_mean = _vis_weights(y, E, mode='mean')
        assert len(w_mean) == len(E)
        assert np.allclose(w_mean, [1.5, 1.5, 2.0, 2.0])
    
    def test_graph_from_edges_weighted(self):
        """Test creation of weighted graph from edges."""
        n = 3
        E = np.array([(0, 1), (1, 2)])
        w = np.array([0.5, 1.0])
        
        # Test sparse output
        G_sparse = _graph_from_edges_weighted(n, E, w, sparse=True)
        assert isinstance(G_sparse, nx.Graph)
        assert G_sparse.number_of_nodes() == n
        assert G_sparse[0][1]['weight'] == 0.5
        assert G_sparse[1][2]['weight'] == 1.0
        
        # Test dense output
        G_dense = _graph_from_edges_weighted(n, E, w, sparse=False)
        assert isinstance(G_dense, nx.Graph)
        assert G_dense.number_of_nodes() == n
        assert G_dense[0][1]['weight'] == 0.5
        assert G_dense[1][2]['weight'] == 1.0
    
    def test_ordinal_patterns(self):
        """Test ordinal pattern extraction."""
        # Test with order 2
        x = np.array([3, 1, 2, 4, 5, 3, 2])
        patterns = _ordinal_patterns(x, order=2)
        assert len(patterns) == len(x) - 2
        assert all(isinstance(p, int) for p in patterns)
        
        # Test with order 3
        patterns = _ordinal_patterns(x, order=3)
        assert len(patterns) == len(x) - 3
        assert all(isinstance(p, int) for p in patterns)
    
    def test_adj_to_graph(self):
        """Test conversion of adjacency matrix to graph."""
        # Test undirected graph
        A_undir = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        G_undir = _adj_to_graph(A_undir, directed=False)
        assert isinstance(G_undir, nx.Graph)
        assert G_undir.number_of_nodes() == 3
        assert G_undir.number_of_edges() == 2
        
        # Test directed graph
        A_dir = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        G_dir = _adj_to_graph(A_dir, directed=True)
        assert isinstance(G_dir, nx.DiGraph)
        assert G_dir.number_of_nodes() == 3
        assert G_dir.number_of_edges() == 3
    
    def test_hvg_edges(self):
        """Test horizontal visibility graph edge detection."""
        # Simple peak
        y = np.array([1, 2, 1])
        edges = _hvg_edges(y)
        assert len(edges) == 2  # (0,1) and (1,2)
        assert (0, 1) in edges
        assert (1, 2) in edges
        
        # Test with visibility
        y = np.array([1, 3, 1, 2])
        edges = _hvg_edges(y)
        assert (0, 2) in edges  # Should be visible
        assert (0, 3) not in edges  # Should be blocked


class TestGraphMetrics:
    """Test cases for graph metric calculations."""
    
    def test_triangle_count(self):
        """Test triangle counting in a graph."""
        # No triangles
        G = nx.cycle_graph(4)
        assert triangle_count(G) == 0
        
        # One triangle
        G.add_edge(0, 2)
        assert triangle_count(G) == 1
        
        # Two triangles
        G.add_edge(1, 3)
        assert triangle_count(G) == 2
    
    def test_wedge_count(self):
        """Test wedge (2-star) counting in a graph."""
        # No wedges
        G = nx.Graph([(0, 1)])
        assert wedge_count(G) == 0
        
        # One wedge
        G.add_edge(1, 2)
        assert wedge_count(G) == 1
        
        # Two wedges
        G.add_edge(1, 3)
        assert wedge_count(G) == 3  # (0,1,2), (0,1,3), (2,1,3)
    
    def test_motif_summary(self):
        """Test graph motif summary."""
        # Simple triangle graph
        G = nx.complete_graph(3)
        summary = motif_summary(G)
        assert summary["n"] == 3
        assert summary["m"] == 3
        assert summary["triangles"] == 1
        assert summary["wedges"] == 3
    
    def test_giant_component(self):
        """Test extraction of giant component."""
        # Two components, one larger than the other
        G = nx.Graph([(0, 1), (1, 2), (3, 4)])
        giant = _giant_component(G)
        assert giant.number_of_nodes() == 3
        assert giant.number_of_edges() == 2
    
    def test_small_world_summary(self):
        """Test small-world summary metrics."""
        # Ring graph
        G = nx.cycle_graph(10)
        summary = small_world_summary(G)
        assert "avg_degree" in summary
        assert "avg_path_length" in summary
        assert "clustering" in summary
        assert 0 <= summary["clustering"] <= 1
    
    def test_directed_3node_motifs(self):
        """Test counting of directed 3-node motifs."""
        # Simple directed triangle
        G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
        counts = directed_3node_motifs(G)
        assert sum(counts.values()) > 0
        
        # Test with sampling
        counts_sampled = directed_3node_motifs(G, max_samples=5)
        assert len(counts_sampled) > 0
    
    def test_undirected_4node_motifs(self):
        """Test counting of undirected 4-node motifs."""
        # Simple 4-clique
        G = nx.complete_graph(4)
        counts = undirected_4node_motifs(G)
        assert sum(counts.values()) > 0
        
        # Test with sampling
        counts_sampled = undirected_4node_motifs(G, max_samples=5)
        assert len(counts_sampled) > 0


class TestGraphSummary:
    """Test cases for graph summary functionality."""
    
    def test_graph_summary_basic(self):
        """Test basic graph summary."""
        G = nx.erdos_renyi_graph(10, 0.3)
        summary = graph_summary(G)
        
        # Check basic metrics
        assert summary["n"] == 10
        assert 0 <= summary["density"] <= 1
        assert "avg_degree" in summary
        assert "avg_path_length" in summary
        
        # Test with motif counting
        summary_with_motifs = graph_summary(G, motifs="3node")
        assert "motif_counts" in summary_with_motifs
        
        # Test with sampling
        summary_sampled = graph_summary(G, motifs="4node", motif_samples=10)
        assert "motif_counts" in summary_sampled


class TestBatchProcessing:
    """Test cases for batch processing functionality."""
    
    def test_batch_transform(self):
        """Test batch transformation of time series."""
        # Create some test time series
        X = [np.random.randn(10) for _ in range(3)]
        
        # Test with a simple transformation (just return the input)
        def identity_builder(series, **kwargs):
            return series
            
        # Monkey-patch the builder lookup for testing
        import ts2net.core
        original_builders = ts2net.core._BUILDERS
        ts2net.core._BUILDERS = {"identity": identity_builder}
        
        try:
            results = batch_transform(X, builder="identity")
            assert len(results) == len(X)
            for x, y in zip(X, results):
                assert np.array_equal(x, y)
        finally:
            # Restore original builders
            ts2net.core._BUILDERS = original_builders
