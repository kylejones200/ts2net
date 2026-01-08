"""
Tests for joint and cross methods for multivariate time series analysis.
"""

import numpy as np
import pytest
import networkx as nx
from ts2net.multivariate.joint_cross import (
    joint_recurrence_network,
    cross_visibility_graph,
    coupling_strength,
    network_comparison_metrics,
)


class TestJointRecurrenceNetwork:
    """Test joint recurrence network construction."""
    
    def test_joint_recurrence_basic(self):
        """Test basic joint recurrence network."""
        x1 = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        x2 = np.array([1.1, 2.1, 1.6, 3.1, 2.6])
        
        G, A = joint_recurrence_network(
            x1, x2,
            threshold=0.5,
            method="epsilon"
        )
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(x1)
        assert A.shape == (len(x1), len(x1))
        assert not G.is_directed()
    
    def test_joint_recurrence_knn(self):
        """Test joint recurrence with k-NN method."""
        x1 = np.random.randn(50)
        x2 = np.random.randn(50)
        
        G, A = joint_recurrence_network(
            x1, x2,
            k=5,
            method="knn"
        )
        
        assert G.number_of_nodes() == len(x1)
        assert G.number_of_edges() > 0
    
    def test_joint_recurrence_weighted(self):
        """Test weighted joint recurrence network."""
        x1 = np.array([1.0, 2.0, 1.5, 3.0])
        x2 = np.array([1.1, 2.1, 1.6, 3.1])
        
        G, A = joint_recurrence_network(
            x1, x2,
            threshold=0.5,
            weighted=True
        )
        
        # Check that edges have weights
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 0
    
    def test_joint_recurrence_directed(self):
        """Test directed joint recurrence network."""
        x1 = np.random.randn(30)
        x2 = np.random.randn(30)
        
        G, A = joint_recurrence_network(
            x1, x2,
            threshold=0.5,
            directed=True
        )
        
        assert isinstance(G, nx.DiGraph)
        assert G.is_directed()
    
    def test_joint_recurrence_different_lengths(self):
        """Test that different length series raise error."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same length"):
            joint_recurrence_network(x1, x2, threshold=0.5)
    
    def test_joint_recurrence_missing_params(self):
        """Test that missing parameters raise errors."""
        x1 = np.random.randn(20)
        x2 = np.random.randn(20)
        
        with pytest.raises(ValueError, match="threshold required"):
            joint_recurrence_network(x1, x2, method="epsilon")
        
        # k-NN has a default k=5, so it should work without explicit k
        G, A = joint_recurrence_network(x1, x2, method="knn")
        assert G.number_of_nodes() == len(x1)


class TestCrossVisibilityGraph:
    """Test cross visibility graph construction."""
    
    def test_cross_visibility_hvg(self):
        """Test cross horizontal visibility graph."""
        x1 = np.array([1.0, 3.0, 2.0, 4.0])
        x2 = np.array([1.5, 2.5, 3.5, 1.0])
        
        G, A = cross_visibility_graph(x1, x2, method="hvg")
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(x1) + len(x2)
        assert A.shape == (len(x1) + len(x2), len(x1) + len(x2))
    
    def test_cross_visibility_nvg(self):
        """Test cross natural visibility graph."""
        x1 = np.random.randn(30)
        x2 = np.random.randn(30)
        
        G, A = cross_visibility_graph(x1, x2, method="nvg")
        
        assert G.number_of_nodes() == len(x1) + len(x2)
        assert G.number_of_edges() >= 0
    
    def test_cross_visibility_different_lengths(self):
        """Test cross visibility with different length series."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.5, 2.5])
        
        G, A = cross_visibility_graph(x1, x2, method="hvg")
        
        # Should work with different lengths
        assert G.number_of_nodes() == len(x1) + len(x2)
    
    def test_cross_visibility_weighted(self):
        """Test weighted cross visibility graph."""
        x1 = np.array([1.0, 3.0, 2.0, 4.0])
        x2 = np.array([1.5, 2.5, 3.5, 1.0])
        
        G, A = cross_visibility_graph(
            x1, x2,
            method="hvg",
            weighted="absdiff"
        )
        
        # Check that edges have weights
        for u, v, data in G.edges(data=True):
            assert 'weight' in data
    
    def test_cross_visibility_with_limit(self):
        """Test cross visibility with temporal limit."""
        x1 = np.random.randn(50)
        x2 = np.random.randn(50)
        
        G1, _ = cross_visibility_graph(x1, x2, method="hvg", limit=None)
        G2, _ = cross_visibility_graph(x1, x2, method="hvg", limit=10)
        
        # With limit, should have fewer or equal edges
        assert G2.number_of_edges() <= G1.number_of_edges()
    
    def test_cross_visibility_invalid_method(self):
        """Test that invalid method raises error."""
        x1 = np.random.randn(20)
        x2 = np.random.randn(20)
        
        with pytest.raises(ValueError, match="Unknown method"):
            cross_visibility_graph(x1, x2, method="invalid")


class TestCouplingStrength:
    """Test coupling strength computation."""
    
    def test_coupling_strength_joint_recurrence(self):
        """Test coupling strength with joint recurrence."""
        # Create coupled series
        x1 = np.random.randn(100)
        x2 = x1 + 0.1 * np.random.randn(100)  # Strongly coupled
        
        metrics = coupling_strength(
            x1, x2,
            method="joint_recurrence",
            threshold=0.5
        )
        
        assert "coupling_strength" in metrics
        assert "joint_recurrence_rate" in metrics
        assert "synchronization" in metrics
        assert "asymmetry" in metrics
        assert 0 <= metrics["coupling_strength"] <= 10  # Reasonable range
        assert 0 <= metrics["joint_recurrence_rate"] <= 1
        assert -1 <= metrics["synchronization"] <= 1
    
    def test_coupling_strength_knn(self):
        """Test coupling strength with k-NN."""
        x1 = np.random.randn(50)
        x2 = x1 + 0.2 * np.random.randn(50)
        
        metrics = coupling_strength(
            x1, x2,
            method="joint_recurrence",
            k=5
        )
        
        assert "coupling_strength" in metrics
        assert metrics["coupling_strength"] >= 0
    
    def test_coupling_strength_cross_visibility(self):
        """Test coupling strength with cross visibility."""
        x1 = np.random.randn(50)
        x2 = x1 + 0.1 * np.random.randn(50)
        
        metrics = coupling_strength(
            x1, x2,
            method="cross_visibility"
        )
        
        assert "coupling_strength" in metrics
        assert "cross_visibility_density" in metrics
        assert 0 <= metrics["coupling_strength"] <= 1
    
    def test_coupling_strength_uncoupled(self):
        """Test coupling strength for uncoupled series."""
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)  # Independent
        
        metrics = coupling_strength(
            x1, x2,
            method="joint_recurrence",
            threshold=0.5
        )
        
        # Coupling should be lower for independent series
        assert metrics["coupling_strength"] >= 0
    
    def test_coupling_strength_different_lengths(self):
        """Test that different length series raise error."""
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same length"):
            coupling_strength(x1, x2, method="joint_recurrence", threshold=0.5)
    
    def test_coupling_strength_invalid_method(self):
        """Test that invalid method raises error."""
        x1 = np.random.randn(20)
        x2 = np.random.randn(20)
        
        with pytest.raises(ValueError, match="Unknown method"):
            coupling_strength(x1, x2, method="invalid")


class TestNetworkComparisonMetrics:
    """Test network comparison metrics."""
    
    def test_network_comparison_basic(self):
        """Test basic network comparison."""
        # Create three similar networks
        G1 = nx.erdos_renyi_graph(20, 0.3, seed=42)
        G2 = nx.erdos_renyi_graph(20, 0.3, seed=43)
        G3 = nx.erdos_renyi_graph(20, 0.2, seed=44)  # Different density
        
        networks = [G1, G2, G3]
        metrics = network_comparison_metrics(networks)
        
        assert "density_similarity" in metrics
        assert "degree_correlation" in metrics
        assert "edge_overlap" in metrics
        assert "structural_similarity" in metrics
        assert "network_names" in metrics
        
        # Check shapes
        assert metrics["density_similarity"].shape == (3, 3)
        assert metrics["degree_correlation"].shape == (3, 3)
        assert metrics["edge_overlap"].shape == (3, 3)
        assert metrics["structural_similarity"].shape == (3, 3)
    
    def test_network_comparison_with_names(self):
        """Test network comparison with custom names."""
        G1 = nx.erdos_renyi_graph(10, 0.3, seed=42)
        G2 = nx.erdos_renyi_graph(10, 0.3, seed=43)
        
        networks = [G1, G2]
        names = ["network_A", "network_B"]
        
        metrics = network_comparison_metrics(networks, names=names)
        
        assert metrics["network_names"] == names
    
    def test_network_comparison_single_network(self):
        """Test comparison with single network."""
        G = nx.erdos_renyi_graph(10, 0.3, seed=42)
        
        metrics = network_comparison_metrics([G])
        
        assert metrics["density_similarity"].shape == (1, 1)
        assert metrics["density_similarity"][0, 0] == 1.0
    
    def test_network_comparison_different_sizes(self):
        """Test comparison with networks of different sizes."""
        G1 = nx.erdos_renyi_graph(10, 0.3, seed=42)
        G2 = nx.erdos_renyi_graph(15, 0.3, seed=43)
        
        # Should work - uses union of all nodes
        metrics = network_comparison_metrics([G1, G2])
        
        assert metrics["density_similarity"].shape == (2, 2)
    
    def test_network_comparison_directed(self):
        """Test comparison with directed networks."""
        G1 = nx.erdos_renyi_graph(10, 0.3, seed=42, directed=True)
        G2 = nx.erdos_renyi_graph(10, 0.3, seed=43, directed=True)
        
        metrics = network_comparison_metrics([G1, G2])
        
        assert metrics["density_similarity"].shape == (2, 2)
    
    def test_network_comparison_name_mismatch(self):
        """Test that name mismatch raises error."""
        G1 = nx.erdos_renyi_graph(10, 0.3, seed=42)
        G2 = nx.erdos_renyi_graph(10, 0.3, seed=43)
        
        with pytest.raises(ValueError, match="Number of names"):
            network_comparison_metrics([G1, G2], names=["only_one_name"])


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_joint_recurrence_empty_series(self):
        """Test joint recurrence with very short series."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.1, 2.1])
        
        G, A = joint_recurrence_network(x1, x2, threshold=0.5)
        
        assert G.number_of_nodes() == 2
        # May have 0 or 1 edge depending on threshold
    
    def test_cross_visibility_empty_series(self):
        """Test cross visibility with very short series."""
        x1 = np.array([1.0])
        x2 = np.array([2.0])
        
        G, A = cross_visibility_graph(x1, x2, method="hvg")
        
        assert G.number_of_nodes() == 2
        # May have 0 or 1 edge
    
    def test_coupling_strength_constant_series(self):
        """Test coupling strength with constant series."""
        x1 = np.ones(50)
        x2 = np.ones(50)
        
        # Should handle constant series gracefully
        metrics = coupling_strength(
            x1, x2,
            method="joint_recurrence",
            threshold=0.1
        )
        
        assert "coupling_strength" in metrics
        assert not np.isnan(metrics["coupling_strength"])

