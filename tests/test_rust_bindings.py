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
    # 1×1 case — single series, self-distance must be 0
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    D = ts2net_rs.cdist_dtw(x)
    assert D.shape == (1, 1)
    assert np.isclose(D[0, 0], 0.0)

    # 2×2 case — off-diagonal must be finite and positive
    # This is the regression test for the usize::MAX overflow bug:
    # previously i + band overflowed to 0 in release mode, leaving every
    # off-diagonal cell at inf.
    x2 = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    D2 = ts2net_rs.cdist_dtw(x2)
    assert D2.shape == (2, 2)
    assert np.isclose(D2[0, 0], 0.0), "self-distance must be 0"
    assert np.isclose(D2[1, 1], 0.0), "self-distance must be 0"
    assert np.isfinite(D2[0, 1]), "off-diagonal must be finite (overflow bug)"
    assert np.isfinite(D2[1, 0]), "off-diagonal must be finite (overflow bug)"
    assert D2[0, 1] > 0.0, "distinct series must have positive distance"
    assert np.allclose(D2, D2.T), "distance matrix must be symmetric"

    # 3×3 case — all pairwise distances finite and symmetric
    x3 = np.array([
        [1.0, 2.0, 3.0, 2.0, 1.0],
        [5.0, 4.0, 3.0, 4.0, 5.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    D3 = ts2net_rs.cdist_dtw(x3)
    assert D3.shape == (3, 3)
    assert np.all(np.isfinite(D3)), "all pairwise distances must be finite"
    assert np.allclose(D3, D3.T), "distance matrix must be symmetric"
    assert np.allclose(np.diag(D3), 0.0), "diagonal must be zero"

    # Band constraint — must still return finite off-diagonal values
    D_band = ts2net_rs.cdist_dtw(x2, band=2)
    assert D_band.shape == (2, 2)
    assert np.isfinite(D_band[0, 1]), "banded DTW must be finite"

    # Known value: DTW([1,2,3], [1,2,3]) == 0
    same = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    D_same = ts2net_rs.cdist_dtw(same)
    assert np.isclose(D_same[0, 1], 0.0), "identical series must have DTW distance 0"


def test_hvg_degrees():
    y = np.array([1.0, 2.0, 1.0, 2.0, 3.0])
    out = ts2net_rs.hvg_degrees(y)
    assert out["n_edges"] == ts2net_rs.hvg_edges(y).shape[0]
    deg = np.asarray(out["degree"])
    assert deg.shape == (len(y),)
    assert int(deg.sum()) == 2 * out["n_edges"]


def test_nvg_degrees_with_limit():
    y = np.sin(np.linspace(0, 4 * np.pi, 80))
    full = ts2net_rs.nvg_degrees(y)
    limited = ts2net_rs.nvg_degrees(y, limit=10)
    assert limited["n_edges"] <= full["n_edges"]


def test_cdist_dtw_rectangular():
    X = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    rect = ts2net_rs.cdist_dtw_rectangular(X[:2], X[1:])
    full = ts2net_rs.cdist_dtw(X)
    np.testing.assert_allclose(rect, full[:2, 1:], rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    test_hvg_edges()
    test_dtw_distance()
