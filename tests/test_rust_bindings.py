import numpy as np
import ts2net_rs


def test_hvg_edges():
    """Test Horizontal Visibility Graph implementation."""
    # Simple test case: a small time series
    y = np.array([1.0, 2.0, 1.0, 2.0])
    edges = ts2net_rs.hvg_edges(y)

    # Expected edges in a horizontal visibility graph
    expected_edges = np.array([[0, 1], [1, 2], [2, 3]])

    # Check that we have the expected number of edges
    assert edges.shape[0] == expected_edges.shape[0]

    # Check that all expected edges are present
    for edge in expected_edges:
        assert np.any(np.all(edges == edge, axis=1)), f"Missing edge: {edge}"

    print("✅ test_hvg_edges passed")


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

    print("✅ test_dtw_distance passed")


if __name__ == "__main__":
    test_hvg_edges()
    test_dtw_distance()
    print("\nAll tests passed!")
