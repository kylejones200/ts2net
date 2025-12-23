"""
Tests for unified graph visualization API.
"""

import numpy as np
import pytest
import networkx as nx

from ts2net.viz import TSGraph, build_visibility_graph, draw_tsgraph


class TestTSGraph:
    """Tests for TSGraph dataclass."""
    
    def test_tsgraph_creation(self):
        """Test creating a TSGraph."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        pos = {0: np.array([0, 0]), 1: np.array([1, 1]), 2: np.array([2, 0])}
        meta = {"method": "test", "n": 3}
        
        tsgraph = TSGraph(graph=G, pos=pos, meta=meta)
        
        assert tsgraph.graph.number_of_nodes() == 3
        assert tsgraph.graph.number_of_edges() == 2
        assert tsgraph.pos is not None
        assert tsgraph.meta["method"] == "test"
    
    def test_tsgraph_without_pos(self):
        """Test TSGraph without positions."""
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1)
        
        tsgraph = TSGraph(graph=G, pos=None, meta={"method": "test"})
        
        assert tsgraph.pos is None
        assert tsgraph.graph.number_of_nodes() == 2


class TestBuildVisibilityGraph:
    """Tests for build_visibility_graph function."""
    
    def test_build_hvg_basic(self):
        """Test building basic HVG."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        assert tsgraph.graph.number_of_nodes() == len(x)
        assert tsgraph.graph.number_of_edges() > 0
        assert tsgraph.pos is not None
        assert tsgraph.meta["kind"] == "hvg"
        assert tsgraph.meta["method"] == "visibility"
    
    def test_build_hvg_directed(self):
        """Test building directed HVG."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=True)
        
        assert isinstance(tsgraph.graph, nx.DiGraph)
        assert tsgraph.meta["directed"] is True
    
    def test_build_hvg_undirected(self):
        """Test building undirected HVG."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        assert isinstance(tsgraph.graph, nx.Graph)
        assert not isinstance(tsgraph.graph, nx.DiGraph)
        assert tsgraph.meta["directed"] is False
    
    def test_build_nvg(self):
        """Test building NVG."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='nvg', directed=False)
        
        assert tsgraph.graph.number_of_nodes() == len(x)
        assert tsgraph.meta["kind"] == "nvg"
    
    def test_weight_modes(self):
        """Test different weight modes."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        
        # Test absdiff
        tsgraph1 = build_visibility_graph(x, kind='hvg', weighted='absdiff')
        assert all('weight' in tsgraph1.graph[u][v] for u, v in tsgraph1.graph.edges())
        
        # Test time_gap
        tsgraph2 = build_visibility_graph(x, kind='hvg', weighted='time_gap')
        assert all('weight' in tsgraph2.graph[u][v] for u, v in tsgraph2.graph.edges())
        
        # Test slope
        tsgraph3 = build_visibility_graph(x, kind='hvg', weighted='slope')
        assert all('weight' in tsgraph3.graph[u][v] for u, v in tsgraph3.graph.edges())
        
        # Test none (no weights)
        tsgraph4 = build_visibility_graph(x, kind='hvg', weighted=False)
        # May or may not have weights depending on implementation
    
    def test_weighted_bool(self):
        """Test weighted=True defaults to absdiff."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', weighted=True)
        
        assert tsgraph.meta["weight_mode"] == "absdiff"
        # Should have weights on edges
        if tsgraph.graph.number_of_edges() > 0:
            edge = list(tsgraph.graph.edges())[0]
            assert 'weight' in tsgraph.graph[edge[0]][edge[1]]
    
    def test_limit_parameter(self):
        """Test limit parameter."""
        np.random.seed(42)
        x = np.random.randn(50)
        
        # Without limit
        tsgraph1 = build_visibility_graph(x, kind='hvg', limit=None)
        edges1 = tsgraph1.graph.number_of_edges()
        
        # With limit
        tsgraph2 = build_visibility_graph(x, kind='hvg', limit=5)
        edges2 = tsgraph2.graph.number_of_edges()
        
        # Limited graph should have fewer or equal edges
        assert edges2 <= edges1
    
    def test_small_series(self):
        """Test with very small series."""
        x = np.array([1.0])
        tsgraph = build_visibility_graph(x, kind='hvg')
        
        assert tsgraph.graph.number_of_nodes() == 1
        assert tsgraph.graph.number_of_edges() == 0
    
    def test_empty_series(self):
        """Test with empty series."""
        x = np.array([])
        tsgraph = build_visibility_graph(x, kind='hvg')
        
        assert tsgraph.graph.number_of_nodes() == 0
    
    def test_node_attributes(self):
        """Test that nodes have correct attributes."""
        x = np.array([1.0, 3.0, 2.0, 4.0])
        tsgraph = build_visibility_graph(x, kind='hvg')
        
        for i in range(len(x)):
            assert tsgraph.graph.nodes[i]['t'] == i
            assert tsgraph.graph.nodes[i]['x'] == float(x[i])
    
    def test_pos_coordinates(self):
        """Test that pos coordinates match (t, x[t])."""
        x = np.array([1.0, 3.0, 2.0, 4.0])
        tsgraph = build_visibility_graph(x, kind='hvg', return_pos=True)
        
        assert tsgraph.pos is not None
        for i in range(len(x)):
            assert i in tsgraph.pos
            assert np.allclose(tsgraph.pos[i], np.array([i, x[i]]))
    
    def test_pos_none(self):
        """Test return_pos=False."""
        x = np.array([1.0, 3.0, 2.0, 4.0])
        tsgraph = build_visibility_graph(x, kind='hvg', return_pos=False)
        
        assert tsgraph.pos is None


class TestDrawTSGraph:
    """Tests for draw_tsgraph function."""
    
    def test_draw_basic(self):
        """Test basic drawing."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        # Should not raise
        fig, ax = draw_tsgraph(tsgraph, show=False)
        assert fig is not None
        assert ax is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)
    
    def test_draw_with_ax(self):
        """Test drawing on existing axes."""
        plt = pytest.importorskip("matplotlib.pyplot")
        
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        fig, ax = plt.subplots()
        fig2, ax2 = draw_tsgraph(tsgraph, ax=ax, show=False)
        
        assert fig is fig2
        assert ax is ax2
        plt.close(fig)
    
    def test_draw_color_by_time(self):
        """Test coloring by time."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        fig, ax = draw_tsgraph(tsgraph, color_by='time', show=False)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)
    
    def test_draw_color_by_degree(self):
        """Test coloring by degree."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        fig, ax = draw_tsgraph(tsgraph, color_by='degree', show=False)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)
    
    def test_draw_color_by_none(self):
        """Test no coloring."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=False)
        
        fig, ax = draw_tsgraph(tsgraph, color_by='none', show=False)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)
    
    def test_draw_directed(self):
        """Test drawing directed graph."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        tsgraph = build_visibility_graph(x, kind='hvg', directed=True)
        
        fig, ax = draw_tsgraph(tsgraph, show=False)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)
    
    def test_draw_without_pos(self):
        """Test drawing graph without positions (should use layout)."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        for i in range(3):
            G.nodes[i]['t'] = i
            G.nodes[i]['x'] = float(i)
        
        tsgraph = TSGraph(graph=G, pos=None, meta={"method": "test"})
        
        fig, ax = draw_tsgraph(tsgraph, show=False)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def test_hvg_nvg_comparison(self):
        """Test comparing HVG and NVG on same series."""
        x = np.sin(np.linspace(0, 4*np.pi, 50))
        
        tsgraph_hvg = build_visibility_graph(x, kind='hvg')
        tsgraph_nvg = build_visibility_graph(x, kind='nvg')
        
        assert tsgraph_hvg.graph.number_of_nodes() == tsgraph_nvg.graph.number_of_nodes()
        # NVG typically has more edges than HVG
        assert tsgraph_nvg.graph.number_of_edges() >= tsgraph_hvg.graph.number_of_edges()
    
    def test_weight_mode_consistency(self):
        """Test that weight modes produce consistent results."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        
        # All should produce same graph structure, different weights
        tsgraph1 = build_visibility_graph(x, kind='hvg', weighted='absdiff')
        tsgraph2 = build_visibility_graph(x, kind='hvg', weighted='time_gap')
        tsgraph3 = build_visibility_graph(x, kind='hvg', weighted='slope')
        
        # Same number of edges
        assert tsgraph1.graph.number_of_edges() == tsgraph2.graph.number_of_edges()
        assert tsgraph2.graph.number_of_edges() == tsgraph3.graph.number_of_edges()
        
        # All should have weights
        for u, v in tsgraph1.graph.edges():
            assert 'weight' in tsgraph1.graph[u][v]
            assert 'weight' in tsgraph2.graph[u][v]
            assert 'weight' in tsgraph3.graph[u][v]
