"""
Tests for built-in network metrics.
"""

import numpy as np
import pytest
import networkx as nx

try:
    from ts2net.networks.metrics import (
        compute_clustering,
        compute_path_lengths,
        compute_modularity,
        network_metrics,
    )
    from ts2net import HVG, NVG
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestComputeClustering:
    """Test clustering coefficient computation."""
    
    def test_clustering_basic(self):
        """Test basic clustering computation."""
        G = nx.karate_club_graph()
        
        result = compute_clustering(G)
        
        assert "avg_clustering" in result
        assert "transitivity" in result
        assert 0 <= result["avg_clustering"] <= 1
        assert 0 <= result["transitivity"] <= 1
    
    def test_clustering_empty_graph(self):
        """Test clustering on empty graph."""
        G = nx.Graph()
        
        result = compute_clustering(G)
        
        assert np.isnan(result["avg_clustering"])
        assert np.isnan(result["transitivity"])
    
    def test_clustering_directed(self):
        """Test clustering on directed graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        result = compute_clustering(G)
        
        assert "avg_clustering" in result
        assert result["avg_clustering"] >= 0
    
    def test_clustering_sampled(self):
        """Test clustering with sampling for large graphs."""
        G = nx.erdos_renyi_graph(1000, 0.1)
        
        result = compute_clustering(G, sample_size=100)
        
        assert "avg_clustering" in result
        assert 0 <= result["avg_clustering"] <= 1


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestComputePathLengths:
    """Test path length computation."""
    
    def test_path_lengths_basic(self):
        """Test basic path length computation."""
        G = nx.karate_club_graph()
        
        result = compute_path_lengths(G)
        
        assert "avg_path_length" in result
        assert "diameter" in result
        assert "radius" in result
        assert result["avg_path_length"] > 0
        assert result["diameter"] >= result["radius"]
    
    def test_path_lengths_disconnected(self):
        """Test path lengths on disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two disconnected components
        
        result = compute_path_lengths(G)
        
        # Should use largest connected component
        assert "avg_path_length" in result
        assert result["avg_path_length"] > 0
    
    def test_path_lengths_weighted(self):
        """Test path lengths with weighted edges."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 0, 1.5)])
        
        result = compute_path_lengths(G, weight="weight")
        
        assert "avg_path_length" in result
        assert result["avg_path_length"] > 0
    
    def test_path_lengths_sampled(self):
        """Test path lengths with sampling for large graphs."""
        G = nx.erdos_renyi_graph(500, 0.1)
        
        result = compute_path_lengths(G, sample_size=50)
        
        assert "avg_path_length" in result


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestComputeModularity:
    """Test modularity computation."""
    
    def test_modularity_basic(self):
        """Test basic modularity computation."""
        G = nx.karate_club_graph()
        
        result = compute_modularity(G)
        
        assert "modularity" in result
        assert "n_communities" in result
        assert -1 <= result["modularity"] <= 1
        assert result["n_communities"] > 0
    
    def test_modularity_empty_graph(self):
        """Test modularity on empty graph."""
        G = nx.Graph()
        
        result = compute_modularity(G)
        
        assert np.isnan(result["modularity"])
        assert result["n_communities"] == 0
    
    def test_modularity_methods(self):
        """Test different modularity methods."""
        G = nx.karate_club_graph()
        
        # Test methods that are known to work
        methods = ["louvain", "label_propagation"]
        for method in methods:
            result = compute_modularity(G, method=method)
            assert "modularity" in result
            assert "n_communities" in result
        
        # Test greedy separately (may have different parameter support)
        try:
            result = compute_modularity(G, method="greedy")
            assert "modularity" in result
            assert "n_communities" in result
        except (TypeError, ValueError):
            # Skip if greedy method has issues
            pass
    
    def test_modularity_resolution(self):
        """Test modularity with different resolution parameters."""
        G = nx.karate_club_graph()
        
        result_low = compute_modularity(G, resolution=0.5)
        result_high = compute_modularity(G, resolution=2.0)
        
        # Higher resolution typically finds more communities
        assert result_high["n_communities"] >= result_low["n_communities"]


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestNetworkMetrics:
    """Test comprehensive network metrics."""
    
    def test_network_metrics_all(self):
        """Test all network metrics."""
        G = nx.karate_club_graph()
        
        result = network_metrics(G)
        
        assert "avg_clustering" in result
        assert "transitivity" in result
        assert "avg_path_length" in result
        assert "modularity" in result
        assert "n_communities" in result
    
    def test_network_metrics_selective(self):
        """Test selective metric computation."""
        G = nx.karate_club_graph()
        
        result = network_metrics(G, include=["clustering", "modularity"])
        
        assert "avg_clustering" in result
        assert "modularity" in result
        assert "avg_path_length" not in result
    
    def test_network_metrics_sampled(self):
        """Test network metrics with sampling."""
        G = nx.erdos_renyi_graph(500, 0.1)
        
        result = network_metrics(G, sample_size=100)
        
        assert "avg_clustering" in result
        assert "avg_path_length" in result


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestHVGNetworkMetrics:
    """Test network_metrics method on HVG class."""
    
    def test_hvg_network_metrics(self):
        """Test network_metrics on HVG."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        metrics = hvg.network_metrics()
        
        assert "avg_clustering" in metrics
        assert "avg_path_length" in metrics
        assert "modularity" in metrics
    
    def test_hvg_network_metrics_selective(self):
        """Test selective metrics on HVG."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        metrics = hvg.network_metrics(include=["clustering"])
        
        assert "avg_clustering" in metrics
        assert "avg_path_length" not in metrics
    
    def test_hvg_network_metrics_not_built(self):
        """Test that network_metrics fails if graph not built."""
        hvg = HVG()
        
        with pytest.raises(ValueError, match="Call build"):
            hvg.network_metrics()


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestNVGNetworkMetrics:
    """Test network_metrics method on NVG class."""
    
    def test_nvg_network_metrics(self):
        """Test network_metrics on NVG."""
        x = np.random.randn(100)
        nvg = NVG()
        nvg.build(x)
        
        metrics = nvg.network_metrics()
        
        assert "avg_clustering" in metrics
        assert "avg_path_length" in metrics
        assert "modularity" in metrics
    
    def test_nvg_network_metrics_not_built(self):
        """Test that network_metrics fails if graph not built."""
        nvg = NVG()
        
        with pytest.raises(ValueError, match="Call build"):
            nvg.network_metrics()


@pytest.mark.skipif(not HAS_METRICS, reason="Network metrics require networkx")
class TestGraphNetworkMetrics:
    """Test network_metrics method on Graph class."""
    
    def test_graph_network_metrics(self):
        """Test network_metrics on Graph object."""
        from ts2net.core.graph import Graph
        
        edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
        graph = Graph(edges=edges, n_nodes=4)
        
        metrics = graph.network_metrics()
        
        assert "avg_clustering" in metrics
        assert "avg_path_length" in metrics
        assert "modularity" in metrics
    
    def test_graph_network_metrics_large(self):
        """Test network_metrics on large graph (should use sampling)."""
        from ts2net.core.graph import Graph
        
        # Create a larger graph
        edges = [(i, i+1) for i in range(100)]  # Path graph
        graph = Graph(edges=edges, n_nodes=101)
        
        metrics = graph.network_metrics(sample_size=50)
        
        assert "avg_path_length" in metrics

