"""Integration tests for visibility graphs with core functionality."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core import (
    batch_transform,
    graph_summary,
    small_world_summary,
)
from ts2net.core.visibility import HVG, NVG

class TestVisibilityGraphsWithCore:
    """Test integration between visibility graphs and core functionality."""
    
    @pytest.fixture
    def sample_time_series(self):
        """Generate sample time series for testing."""
        # Simple sine wave
        t = np.linspace(0, 4*np.pi, 100)
        return np.sin(t)
    
    def test_hvg_with_graph_summary(self, sample_time_series):
        """Test HVG generation and graph summary."""
        # Create HVG
        hvg = HVG()
        G, A = hvg.fit_transform(sample_time_series)
        
        # Generate graph summary
        summary = graph_summary(G)
        
        # Basic graph properties
        assert summary["n"] == len(sample_time_series)
        assert summary["m"] > 0
        assert 0 <= summary["density"] <= 1
        
        # Check small-world properties
        sw_summary = small_world_summary(G)
        assert "C" in sw_summary  # Clustering coefficient
        assert "L" in sw_summary  # Average path length
    
    def test_nvg_with_graph_summary(self, sample_time_series):
        """Test NVG generation and graph summary."""
        # Create NVG
        nvg = NVG()
        G, A = nvg.fit_transform(sample_time_series)
        
        # Generate graph summary
        summary = graph_summary(G)
        
        # Basic graph properties
        assert summary["n"] == len(sample_time_series)
        assert summary["m"] > 0
        assert 0 <= summary["density"] <= 1
        
        # Check small-world properties
        sw_summary = small_world_summary(G)
        assert "C" in sw_summary  # Clustering coefficient
        assert "L" in sw_summary  # Average path length
    
    def test_batch_transform_with_visibility(self, sample_time_series):
        """Test batch transform with visibility graph builder."""
        # Create multiple time series
        X = [sample_time_series, sample_time_series * -1, sample_time_series * 2]
        
        # Define a builder that uses HVG
        def hvg_builder(series, **kwargs):
            hvg = HVG(**kwargs)
            G, A = hvg.fit_transform(series)
            G.graph['adjacency'] = A
            return G
        
        # Register the builder
        import ts2net.core
        original_builders = ts2net.core._BUILDERS
        ts2net.core._BUILDERS = {"hvg_test": hvg_builder}
        
        try:
            # Process all time series
            results = batch_transform(X, builder="hvg_test")
            
            # Check results
            assert len(results) == len(X)
            for G in results:
                assert isinstance(G, nx.Graph)
                assert 'adjacency' in G.graph
                assert G.number_of_nodes() == len(sample_time_series)
                assert G.number_of_edges() > 0
                
        finally:
            # Restore original builders
            ts2net.core._BUILDERS = original_builders
    
    def test_visibility_graph_properties(self, sample_time_series):
        """Test properties specific to visibility graphs."""
        # Create both HVG and NVG
        hvg = HVG()
        nvg = NVG()
        
        G_hvg, A_hvg = hvg.fit_transform(sample_time_series)
        G_nvg, A_nvg = nvg.fit_transform(sample_time_series)
        
        # HVG is always a subgraph of NVG
        assert G_hvg.number_of_edges() <= G_nvg.number_of_edges()
        
        # Check that all HVG edges are in NVG
        for u, v in G_hvg.edges():
            assert G_nvg.has_edge(u, v)
        
        # Check degree distribution properties
        degrees_hvg = [d for n, d in G_hvg.degree()]
        degrees_nvg = [d for n, d in G_nvg.degree()]
        
        assert np.mean(degrees_hvg) <= np.mean(degrees_nvg)
        assert np.max(degrees_hvg) <= np.max(degrees_nvg)
