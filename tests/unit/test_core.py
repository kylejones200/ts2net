"""Simplified unit tests for core module functionality."""
import numpy as np
import networkx as nx
from ts2net.core import graph_summary, batch_transform


class TestGraphSummary:
    """Basic tests for graph summary functionality."""
    
    def test_graph_summary_basic(self):
        """Test basic graph summary."""
        G = nx.erdos_renyi_graph(10, 0.3)
        summary = graph_summary(G)
        
        # Check basic metrics
        assert summary["n"] == 10
        assert 0 <= summary["density"] <= 1
        assert "deg_mean" in summary


class TestBatchProcessing:
    """Basic tests for batch processing functionality."""
    
    def test_batch_transform(self):
        """Test batch transformation of time series."""
        np.random.seed(42)
        X = [np.random.randn(10) for _ in range(3)]
        results = batch_transform(X, builder="hvg")
        assert len(results) == len(X)
        assert all(isinstance(g, nx.Graph) for g in results)
