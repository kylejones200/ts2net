"""Simplified unit tests for visibility graph implementations."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core.visibility import HVG, NVG


class TestHVG:
    """Basic tests for Horizontal Visibility Graph (HVG)."""
    
    def test_hvg_basic(self):
        """Test HVG on a simple time series."""
        x = np.array([1, 2, 3, 2, 1])
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() > 0
        assert nx.is_connected(G)


class TestNVG:
    """Basic tests for Natural Visibility Graph (NVG)."""
    
    def test_nvg_basic(self):
        """Test NVG on a simple time series."""
        x = np.array([1, 2, 3, 2, 1])
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() > 0
        assert nx.is_connected(G)


class TestVisibilityGraphCommon:
    """Common basic tests for both HVG and NVG."""
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_empty_series(self, graph_class):
        """Test with empty input series."""
        x = np.array([])
        G, A = graph_class().fit_transform(x)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
