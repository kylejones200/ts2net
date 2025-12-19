"""
Tests for extended multivariate functionality:
- VOI, MIC, van Rossum distances
- Windowing utilities
- Approximate k-NN
"""

import pytest
import numpy as np
import networkx as nx
from ts2net.multivariate.distances import tsdist_voi, tsdist_vr
from ts2net.multivariate.windows import (
    ts_to_windows, ts_to_windows_list, ts_to_windows_labeled, ts_window_stats
)

# Check if optional dependencies are available
try:
    from ts2net.multivariate.distances import tsdist_mic
    HAS_MINEPY = True
except ImportError:
    HAS_MINEPY = False

try:
    from ts2net.multivariate.builders import net_knn_approx, net_enn_approx
    HAS_PYNNDESCENT = True
except ImportError:
    HAS_PYNNDESCENT = False


# ============================================================================
# Distance Function Tests
# ============================================================================

def test_tsdist_voi():
    """Test Variation of Information distance"""
    x = np.random.randn(100)
    y = x + 0.1 * np.random.randn(100)  # Similar to x
    
    d = tsdist_voi(x, y, bins=10)
    assert d >= 0
    assert d < 2.0  # Should be small for similar signals


def test_tsdist_voi_identical():
    """Test VOI with identical series"""
    x = np.random.randn(100)
    d = tsdist_voi(x, x, bins=10)
    assert d < 0.01  # Should be near 0


@pytest.mark.skipif(not HAS_MINEPY, reason="minepy not installed")
def test_tsdist_mic():
    """Test Maximal Information Coefficient distance"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)  # Nonlinear relationship
    
    d = tsdist_mic(x, y, alpha=0.6, c=15)
    assert 0 <= d <= 1
    assert d < 0.5  # Should detect strong association


@pytest.mark.skipif(not HAS_MINEPY, reason="minepy not installed")
def test_tsdist_mic_independent():
    """Test MIC with independent series"""
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    
    d = tsdist_mic(x, y)
    assert 0 <= d <= 1
    assert d > 0.5  # Should be high for independent signals


def test_tsdist_vr():
    """Test van Rossum distance"""
    x = np.zeros(100)
    x[[10, 30, 50, 70, 90]] = 1.0  # Spikes
    
    y = np.zeros(100)
    y[[11, 31, 51, 71, 91]] = 1.0  # Slightly shifted spikes
    
    d = tsdist_vr(x, y, tau=2.0, threshold=0.5)
    assert 0 <= d <= 1
    assert d < 0.5  # Should be relatively small for similar spike trains


def test_tsdist_vr_identical():
    """Test van Rossum with identical spike trains"""
    x = np.zeros(100)
    x[[10, 30, 50]] = 1.0
    
    d = tsdist_vr(x, x, tau=1.0)
    assert d < 0.01  # Should be near 0


# ============================================================================
# Windowing Tests
# ============================================================================

def test_ts_to_windows_basic():
    """Test basic window extraction"""
    x = np.arange(100, dtype=float)
    windows = ts_to_windows(x, width=10, by=1)
    
    assert windows.shape == (91, 10)
    assert np.allclose(windows[0], np.arange(10))
    assert np.allclose(windows[1], np.arange(1, 11))


def test_ts_to_windows_step():
    """Test window extraction with larger step"""
    x = np.arange(100, dtype=float)
    windows = ts_to_windows(x, width=10, by=5)
    
    assert windows.shape == (19, 10)
    assert np.allclose(windows[0], np.arange(10))
    assert np.allclose(windows[1], np.arange(5, 15))


def test_ts_to_windows_start_end():
    """Test window extraction with start/end"""
    x = np.arange(100, dtype=float)
    windows = ts_to_windows(x, width=10, by=1, start=10, end=50)
    
    assert windows.shape == (31, 10)
    assert np.allclose(windows[0], np.arange(10, 20))


def test_ts_to_windows_invalid():
    """Test error handling for invalid parameters"""
    x = np.arange(100, dtype=float)
    
    with pytest.raises(ValueError, match="width must be positive"):
        ts_to_windows(x, width=0, by=1)
    
    with pytest.raises(ValueError, match="by must be positive"):
        ts_to_windows(x, width=10, by=0)
    
    with pytest.raises(ValueError, match="width.*cannot exceed"):
        ts_to_windows(x, width=101, by=1)


def test_ts_to_windows_list():
    """Test windowing multiple series"""
    series_list = [np.random.randn(100) for _ in range(3)]
    windows = ts_to_windows_list(series_list, width=10, by=5)
    
    # Each series produces 19 windows
    assert windows.shape == (3 * 19, 10)


def test_ts_to_windows_labeled():
    """Test windowing with labels"""
    x = np.arange(100, dtype=float)
    windows, indices = ts_to_windows_labeled(x, width=10, by=5)
    
    assert windows.shape[0] == len(indices)
    assert np.array_equal(indices, np.arange(0, 91, 5))


def test_ts_window_stats():
    """Test window statistics calculation"""
    x = np.sin(np.linspace(0, 4*np.pi, 100))
    windows = ts_to_windows(x, width=10, by=1)
    stats = ts_window_stats(windows)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'trend' in stats
    assert len(stats['mean']) == windows.shape[0]


def test_windowing_integration():
    """Test complete windowing → distance → network pipeline"""
    from ts2net.multivariate import ts_dist, net_enn
    
    # Create time series with periodic pattern
    x = np.sin(np.linspace(0, 8*np.pi, 200)) + 0.1 * np.random.randn(200)
    
    # Extract windows
    windows = ts_to_windows(x, width=20, by=10)
    
    # Calculate distances
    D = ts_dist(windows, method='correlation', n_jobs=1)
    
    # Build network
    G, A = net_enn(D, percentile=30)
    
    assert G.number_of_nodes() == windows.shape[0]
    assert G.number_of_edges() > 0


# ============================================================================
# Approximate k-NN Tests
# ============================================================================

@pytest.mark.skipif(not HAS_PYNNDESCENT, reason="pynndescent not installed")
def test_net_knn_approx():
    """Test approximate k-NN construction"""
    np.random.seed(42)
    # Use feature matrix instead of distance matrix (pynndescent works better with features)
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    
    G, A = net_knn_approx(X, k=5, metric='euclidean', weighted=False)
    
    assert G.number_of_nodes() == 100
    # Approximate, so edges may vary - just check we have edges
    assert G.number_of_edges() > 0
    assert G.number_of_edges() <= 1000  # Reasonable upper bound


@pytest.mark.skipif(not HAS_PYNNDESCENT, reason="pynndescent not installed")
def test_net_knn_approx_weighted():
    """Test weighted approximate k-NN"""
    np.random.seed(42)
    # Use feature matrix instead of distance matrix
    X = np.random.rand(50, 10)  # 50 samples, 10 features
    
    G, A = net_knn_approx(X, k=3, metric='euclidean', weighted=True)
    
    assert G.number_of_nodes() == 50
    # Check that edges have weights
    for u, v, data in G.edges(data=True):
        assert 'weight' in data
        assert data['weight'] > 0


@pytest.mark.skipif(not HAS_PYNNDESCENT, reason="pynndescent not installed")
def test_net_enn_approx():
    """Test approximate ε-NN construction"""
    np.random.seed(42)
    # Use feature matrix instead of distance matrix
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    
    G, A = net_enn_approx(X, percentile=20, metric='euclidean', n_neighbors=min(30, X.shape[0]-1))
    
    assert G.number_of_nodes() == 100
    assert G.number_of_edges() > 0


@pytest.mark.skipif(not HAS_PYNNDESCENT, reason="pynndescent not installed")
def test_approx_vs_exact():
    """Compare approximate vs exact k-NN (small dataset)"""
    from ts2net.multivariate import net_knn
    
    np.random.seed(42)
    # Use feature matrix for approximate, distance matrix for exact
    X = np.random.rand(30, 10)  # 30 samples, 10 features
    # For exact, compute distance matrix
    from ts2net.multivariate import ts_dist
    D = ts_dist(X, method='correlation', n_jobs=1)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    # Exact
    G_exact, A_exact = net_knn(D, k=3, weighted=False)
    
    # Approximate (use feature matrix)
    G_approx, A_approx = net_knn_approx(X, k=3, metric='euclidean', 
                                         n_neighbors=10, weighted=False)
    
    # Should be similar (but not necessarily identical)
    assert abs(G_exact.number_of_edges() - G_approx.number_of_edges()) < 10


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_extended_pipeline():
    """Test complete pipeline with new features"""
    from ts2net.multivariate import ts_dist, net_knn
    
    np.random.seed(42)
    
    # Generate time series with structure
    n_series = 20
    n_points = 100
    X = []
    
    for i in range(n_series):
        if i < 10:
            # Group 1: sine waves
            x = np.sin(np.linspace(0, 4*np.pi, n_points)) + 0.1 * np.random.randn(n_points)
        else:
            # Group 2: cosine waves
            x = np.cos(np.linspace(0, 4*np.pi, n_points)) + 0.1 * np.random.randn(n_points)
        X.append(x)
    
    X = np.array(X)
    
    # Calculate distance with VOI
    D = ts_dist(X, method='voi', bins=10, n_jobs=1)
    
    # Build network
    G, A = net_knn(D, k=3, weighted=True)
    
    # Analyze
    assert G.number_of_nodes() == n_series
    assert G.number_of_edges() > 0
    
    # Check clustering (should detect two groups)
    clustering = nx.average_clustering(G)
    assert clustering > 0


def test_windowing_proximity_network():
    """Test proximity network construction from windows (R ts2net style)"""
    from ts2net.multivariate import ts_to_windows, ts_dist, net_enn
    
    # CO2-like data (increasing trend with seasonal oscillation)
    t = np.linspace(0, 10, 200)
    co2 = 300 + 2*t + 10*np.sin(2*np.pi*t) + np.random.randn(200) * 0.5
    
    # Extract windows (R: ts_to_windows)
    windows = ts_to_windows(co2, width=12, by=1)
    
    # Distance matrix (R: ts_dist)
    D = ts_dist(windows, method='correlation', n_jobs=1)
    
    # Network construction (R: net_enn)
    G, A = net_enn(D, percentile=25)
    
    assert G.number_of_nodes() == windows.shape[0]
    assert G.number_of_edges() > 0
    
    # Should form connected temporal structure
    assert nx.is_connected(G) or len(list(nx.connected_components(G))) < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

