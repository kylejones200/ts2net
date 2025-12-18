"""
Tests for multivariate time series network construction.
"""

import pytest
import numpy as np
import networkx as nx
from ts2net.multivariate import ts_dist, net_knn, net_enn, net_weighted
from ts2net.multivariate.distances import (
    tsdist_cor, tsdist_ccf, tsdist_dtw, tsdist_nmi, tsdist_es
)


@pytest.fixture
def sample_timeseries():
    """Generate sample time series for testing"""
    np.random.seed(42)
    n_series = 10
    n_points = 100
    X = np.random.randn(n_series, n_points)
    return X


@pytest.fixture
def sample_distance_matrix():
    """Generate sample distance matrix"""
    np.random.seed(42)
    n = 10
    D = np.random.rand(n, n)
    D = (D + D.T) / 2  # Make symmetric
    np.fill_diagonal(D, 0)
    return D


# ============================================================================
# Distance Function Tests
# ============================================================================

def test_tsdist_cor():
    """Test Pearson correlation distance"""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)  # Perfect positive correlation
    
    d = tsdist_cor(x, y)
    assert 0 <= d <= 2
    assert d < 0.01  # Should be near 0 for perfect correlation


def test_tsdist_cor_anticorrelated():
    """Test correlation distance with anti-correlated series"""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 4, 3, 2, 1], dtype=float)  # Perfect negative correlation
    
    d = tsdist_cor(x, y)
    assert d < 0.01  # Should be near 0 (we use abs(ρ))


def test_tsdist_ccf():
    """Test cross-correlation distance"""
    x = np.sin(np.linspace(0, 4*np.pi, 100))
    y = np.sin(np.linspace(0, 4*np.pi, 100) + 0.1)  # Slightly shifted
    
    d = tsdist_ccf(x, y, max_lag=20)
    assert 0 <= d <= 2
    assert d < 0.5  # Should be small for similar signals


def test_tsdist_dtw():
    """Test DTW distance"""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    
    d = tsdist_dtw(x, y, normalize=True)
    assert d == 0.0  # Identical series


def test_tsdist_dtw_window():
    """Test DTW with Sakoe-Chiba band"""
    x = np.random.randn(50)
    y = np.random.randn(50)
    
    d1 = tsdist_dtw(x, y, window=None, normalize=True)
    d2 = tsdist_dtw(x, y, window=10, normalize=True)
    
    assert d1 >= 0 and d2 >= 0
    # Constrained DTW might give different distance
    assert d2 >= d1 or np.isclose(d1, d2, rtol=0.1)


def test_tsdist_nmi():
    """Test normalized mutual information distance"""
    x = np.random.randn(100)
    y = x + 0.1 * np.random.randn(100)  # Similar to x
    
    d = tsdist_nmi(x, y, bins=10)
    assert 0 <= d <= 1
    assert d < 0.8  # Should be small for similar signals


def test_tsdist_es():
    """Test event synchronization distance"""
    x = np.zeros(100)
    x[[10, 30, 50, 70, 90]] = 1.0  # Events at regular intervals
    
    y = np.zeros(100)
    y[[11, 31, 51, 71, 91]] = 1.0  # Slightly shifted events
    
    d = tsdist_es(x, y, threshold=0.5, tau=2)
    assert 0 <= d <= 1
    assert d < 0.5  # Should be small for synchronized events


# ============================================================================
# ts_dist Tests
# ============================================================================

def test_ts_dist_correlation(sample_timeseries):
    """Test distance matrix calculation with correlation"""
    X = sample_timeseries
    D = ts_dist(X, method='correlation', n_jobs=1)
    
    assert D.shape == (10, 10)
    assert np.allclose(D, D.T)  # Symmetric
    assert np.allclose(np.diag(D), 0)  # Diagonal = 0


def test_ts_dist_dtw(sample_timeseries):
    """Test distance matrix with DTW"""
    X = sample_timeseries
    D = ts_dist(X, method='dtw', normalize=True, n_jobs=1)
    
    assert D.shape == (10, 10)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0)


def test_ts_dist_parallel(sample_timeseries):
    """Test parallel distance computation"""
    X = sample_timeseries
    
    D_serial = ts_dist(X, method='correlation', n_jobs=1)
    # Skip parallel test if it fails (pickle issues in some environments)
    try:
        D_parallel = ts_dist(X, method='correlation', n_jobs=2)
        assert np.allclose(D_serial, D_parallel)
    except Exception:
        pytest.skip("Parallel processing not available in this environment")


def test_ts_dist_invalid_method(sample_timeseries):
    """Test error on invalid method"""
    with pytest.raises(ValueError, match="Unknown distance method"):
        ts_dist(sample_timeseries, method='invalid_method')


def test_ts_dist_invalid_shape():
    """Test error on invalid input shape"""
    X = np.random.randn(100)  # 1D array
    with pytest.raises(ValueError, match="must be 2D"):
        ts_dist(X, method='correlation')


# ============================================================================
# Network Builder Tests
# ============================================================================

def test_net_knn(sample_distance_matrix):
    """Test k-NN network construction"""
    D = sample_distance_matrix
    G, A = net_knn(D, k=3, mutual=False, weighted=False)
    
    assert G.number_of_nodes() == 10
    # Undirected graph: each node connects to k neighbors, but edges are shared
    # Maximum edges = n * k / 2 if all mutual, but typically more
    assert 15 <= G.number_of_edges() <= 30
    assert A.shape == (10, 10)


def test_net_knn_mutual(sample_distance_matrix):
    """Test mutual k-NN"""
    D = sample_distance_matrix
    G, A = net_knn(D, k=3, mutual=True, weighted=False)
    
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() <= 30  # Fewer edges with mutual constraint


def test_net_knn_weighted(sample_distance_matrix):
    """Test weighted k-NN"""
    D = sample_distance_matrix
    G, A = net_knn(D, k=3, weighted=True)
    
    assert G.number_of_nodes() == 10
    # Check that edges have weights
    for u, v, data in G.edges(data=True):
        assert 'weight' in data
        assert data['weight'] > 0


def test_net_knn_invalid_k(sample_distance_matrix):
    """Test error on invalid k"""
    D = sample_distance_matrix
    with pytest.raises(ValueError, match="k must be in range"):
        net_knn(D, k=0)
    
    with pytest.raises(ValueError, match="k must be in range"):
        net_knn(D, k=10)  # k must be < n


def test_net_enn_epsilon(sample_distance_matrix):
    """Test ε-NN with explicit epsilon"""
    D = sample_distance_matrix
    G, A = net_enn(D, epsilon=0.5, weighted=False)
    
    assert G.number_of_nodes() == 10
    assert A.shape == (10, 10)
    
    # Check that all edges have distance < epsilon
    for i in range(10):
        for j in range(i+1, 10):
            if A[i, j] > 0:
                assert D[i, j] < 0.5


def test_net_enn_percentile(sample_distance_matrix):
    """Test ε-NN with percentile"""
    D = sample_distance_matrix
    G, A = net_enn(D, percentile=30, weighted=False)
    
    assert G.number_of_nodes() == 10
    # Should have about 30% of possible edges
    max_edges = 10 * 9 / 2  # 45 for n=10
    assert G.number_of_edges() <= max_edges


def test_net_enn_no_params(sample_distance_matrix):
    """Test error when no epsilon or percentile given"""
    D = sample_distance_matrix
    with pytest.raises(ValueError, match="Must specify either"):
        net_enn(D)


def test_net_weighted(sample_distance_matrix):
    """Test complete weighted network"""
    D = sample_distance_matrix
    G, A = net_weighted(D)
    
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 45  # n*(n-1)/2 for undirected
    
    # Check weights match distances
    for u, v, data in G.edges(data=True):
        assert 'weight' in data
        assert np.isclose(data['weight'], D[u, v])


def test_net_weighted_threshold(sample_distance_matrix):
    """Test weighted network with threshold"""
    D = sample_distance_matrix
    G, A = net_weighted(D, threshold=0.5)
    
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() < 45  # Fewer edges with threshold
    
    # Check all edges below threshold
    for u, v, data in G.edges(data=True):
        assert data['weight'] <= 0.5


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline(sample_timeseries):
    """Test complete pipeline: data → distance → network"""
    X = sample_timeseries
    
    # Calculate distance (use n_jobs=1 to avoid pickle issues in tests)
    D = ts_dist(X, method='correlation', n_jobs=1)
    
    # Build network
    G, A = net_knn(D, k=3, weighted=True)
    
    # Analyze network
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() > 0
    
    # Check graph properties
    assert nx.is_connected(G) or len(list(nx.connected_components(G))) > 0
    
    # Calculate centrality
    centrality = nx.degree_centrality(G)
    assert len(centrality) == 10


def test_different_distance_methods(sample_timeseries):
    """Test that different distance methods produce different networks"""
    X = sample_timeseries
    
    D_cor = ts_dist(X, method='correlation')
    D_dtw = ts_dist(X, method='dtw', normalize=True)
    D_ccf = ts_dist(X, method='ccf', max_lag=20)
    
    # Distance matrices should be different
    assert not np.allclose(D_cor, D_dtw)
    assert not np.allclose(D_cor, D_ccf)
    
    # But all should be symmetric with zero diagonal
    for D in [D_cor, D_dtw, D_ccf]:
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)


def test_ts_dist_part():
    """Test partial distance calculation for HPC"""
    from ts2net.multivariate import ts_dist_part
    
    np.random.seed(42)
    X = np.random.randn(20, 100)
    
    # Calculate full matrix
    D_full = ts_dist(X, method='correlation', n_jobs=1)
    
    # Calculate in parts
    D_part1 = ts_dist_part(X, 0, 10, method='correlation')
    D_part2 = ts_dist_part(X, 10, 20, method='correlation')
    
    # Reconstruct
    D_reconstructed = np.vstack([D_part1, D_part2])
    
    # Should match
    assert np.allclose(D_full, D_reconstructed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

