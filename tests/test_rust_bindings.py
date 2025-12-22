import numpy as np
import ts2net_rs


def test_hvg_edges():
    """Test Horizontal Visibility Graph implementation."""
    # Simple test case: a small time series
    y = np.array([1.0, 2.0, 1.0, 2.0])
    edges = ts2net_rs.hvg_edges(y)

    # Check that we get edges (at least adjacent ones)
    assert edges.shape[0] >= 3  # At least adjacent edges
    assert edges.shape[1] == 2  # Each edge has 2 nodes
    
    # Check that all edges are valid (nodes within range)
    assert np.all(edges >= 0)
    assert np.all(edges < len(y))

    # Test passed


def test_dtw_distance():
    """Test Dynamic Time Warping distance calculation."""
    # Two simple time series
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[1.0, 1.0, 4.0, 4.0]])

    # Calculate distance matrix
    dist_matrix = ts2net_rs.cdist_dtw(x)

    # Should be a 1x1 matrix with the DTW distance
    assert dist_matrix.shape == (1, 1)
    assert np.isclose(dist_matrix[0, 0], 0.0)  # Should be 0 for identical series

    # Test with two different series
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 4.0]])
    dist_matrix = ts2net_rs.cdist_dtw(x)
    assert dist_matrix.shape == (2, 2)
    assert np.allclose(dist_matrix, dist_matrix.T)  # Should be symmetric

    # Test passed


if __name__ == "__main__":
    test_hvg_edges()
    test_dtw_distance()
