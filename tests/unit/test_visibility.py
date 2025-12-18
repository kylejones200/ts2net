"""Unit tests for visibility graph implementations."""
import numpy as np
import networkx as nx
import pytest
from ts2net.core.visibility import HVG, NVG

class TestHVG:
    """Test cases for Horizontal Visibility Graph (HVG)."""
    
    def test_hvg_monotone_increasing(self):
        """Test HVG on a strictly increasing time series."""
        x = np.arange(10.0)  # Strictly increasing
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        # In a strictly increasing series, only adjacent points are visible
        expected_edges = [(i, i+1) for i in range(len(x)-1)]
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == len(x) - 1
        assert sorted(G.edges()) == expected_edges
        
        # Check degree sequence (all nodes should have degree 2 except endpoints)
        degrees = [d for _, d in G.degree()]
        assert set(degrees) == {1, 2}
        assert degrees.count(1) == 2  # First and last nodes have degree 1
        
    def test_hvg_constant_series(self):
        """Test HVG on a constant time series."""
        x = np.ones(10)  # All values are the same
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        # In a constant series, all points should be connected to their immediate neighbors
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == len(x) - 1
        assert nx.is_connected(G)
        
    def test_hvg_single_peak(self):
        """Test HVG with a single peak."""
        x = np.array([1, 2, 3, 2, 1])
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        # Expected edges: (0,1), (1,2), (2,3), (3,4), (0,4), (1,3)
        expected_edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == 6  # 5 from line + 1 cross edge
        assert sorted(G.edges()) == sorted(expected_edges)
        
    def test_hvg_random_series(self):
        """Test HVG on a random time series."""
        np.random.seed(42)
        x = np.random.randn(20)
        hvg = HVG()
        G, A = hvg.fit_transform(x)
        
        # Basic graph properties
        assert G.number_of_nodes() == len(x)
        assert nx.is_connected(G)
        
        # Check that the adjacency matrix is symmetric
        assert np.allclose(A, A.T)
        
        # Check that the graph is planar (HVG is always planar)
        assert nx.check_planarity(G)[0]


class TestNVG:
    """Test cases for Natural Visibility Graph (NVG)."""
    
    def test_nvg_monotone_increasing(self):
        """Test NVG on a strictly increasing time series."""
        x = np.arange(10.0)  # Strictly increasing
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        # In a strictly increasing series, all points are visible to each other
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == (len(x) * (len(x) - 1)) // 2
        assert nx.is_connected(G)
        
    def test_nvg_constant_series(self):
        """Test NVG on a constant time series."""
        x = np.ones(10)  # All values are the same
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        # In a constant series, all points should be connected to their immediate neighbors
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == len(x) - 1
        assert nx.is_connected(G)
        
    def test_nvg_single_peak(self):
        """Test NVG with a single peak."""
        x = np.array([1, 2, 3, 2, 1])
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        # In NVG, the peak should be visible to all other points
        expected_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3),
            (2, 3), (2, 4),
            (3, 4)
        ]
        assert G.number_of_nodes() == len(x)
        assert G.number_of_edges() == len(expected_edges)
        assert sorted(G.edges()) == sorted(expected_edges)
        
    def test_nvg_random_series(self):
        """Test NVG on a random time series."""
        np.random.seed(42)
        x = np.random.randn(10)  # Smaller size for faster tests
        nvg = NVG()
        G, A = nvg.fit_transform(x)
        
        # Basic graph properties
        assert G.number_of_nodes() == len(x)
        assert nx.is_connected(G)
        
        # Check that the adjacency matrix is symmetric
        assert np.allclose(A, A.T)
        
        # Check that the graph is connected (NVG is always connected)
        assert nx.is_connected(G)
        
        # Check that the graph has the expected number of edges
        # (exact number depends on the random series, but should be at least n-1)
        assert G.number_of_edges() >= len(x) - 1


class TestVisibilityGraphCommon:
    """Common test cases for both HVG and NVG."""
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_empty_series(self, graph_class):
        """Test with empty input series."""
        x = np.array([])
        G, A = graph_class().fit_transform(x)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        assert A.shape == (0, 0)
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_single_point(self, graph_class):
        """Test with a single point."""
        x = np.array([1.0])
        G, A = graph_class().fit_transform(x)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0
        assert A.shape == (1, 1)
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_two_points(self, graph_class):
        """Test with two points."""
        x = np.array([1.0, 2.0])
        G, A = graph_class().fit_transform(x)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert (0, 1) in G.edges()
        assert A.shape == (2, 2)
        assert A[0, 1] == A[1, 0] == 1  # Symmetric
    
    @pytest.mark.parametrize("graph_class", [HVG, NVG])
    def test_invalid_input(self, graph_class):
        """Test with invalid input."""
        with pytest.raises(ValueError):
            graph_class().fit_transform("not a numpy array")
        
        with pytest.raises(ValueError):
            graph_class().fit_transform(np.array([[1, 2], [3, 4]]))  # 2D array
