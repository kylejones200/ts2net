"""Unit tests for distance metrics."""
import numpy as np
import pytest
from ts2net.distances.core import (
    tsdist_cor,
    tsdist_ccf,
    tsdist_dtw,
    tsdist_nmi,
    tsdist_voi,
    tsdist_mic,
    tsdist_vr,
    dist_percentile,
    dist_matrix_normalize,
)

class TestCorrelationDistance:
    """Test cases for correlation-based distance metrics."""
    
    def test_tsdist_cor_pearson(self):
        """Test Pearson correlation distance."""
        # Perfect positive correlation
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([2, 4, 6, 8, 10])
        X = np.vstack([x1, x2])
        D = tsdist_cor(X, method="pearson")
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))
        
        # Perfect negative correlation
        x3 = -x1
        X = np.vstack([x1, x3])
        D = tsdist_cor(X, method="pearson")
        assert np.allclose(D, np.array([[0, 4], [4, 0]]))
        
        # Test with absolute=True
        D_abs = tsdist_cor(X, method="pearson", absolute=True)
        assert np.allclose(D_abs, np.array([[0, 0], [0, 0]]))
    
    def test_tsdist_cor_spearman(self):
        """Test Spearman rank correlation distance."""
        # Perfect monotonic relationship
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([1, 4, 9, 16, 25])  # Quadratic relationship
        X = np.vstack([x1, x2])
        D = tsdist_cor(X, method="spearman")
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))
    
    def test_tsdist_cor_invalid_method(self):
        """Test with invalid correlation method."""
        X = np.random.randn(2, 5)
        with pytest.raises(ValueError):
            tsdist_cor(X, method="invalid")


class TestCCFDistance:
    """Test cases for cross-correlation based distance."""
    
    def test_tsdist_ccf_identical_series(self):
        """Test CCF distance with identical time series."""
        x = np.array([1, 2, 3, 2, 1])
        X = np.vstack([x, x])
        D = tsdist_ccf(X, max_lag=2)
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))
    
    def test_tsdist_ccf_shifted_series(self):
        """Test CCF distance with shifted time series."""
        x1 = np.array([1, 2, 3, 2, 1, 0, 0])
        x2 = np.array([0, 0, 1, 2, 3, 2, 1])
        X = np.vstack([x1, x2])
        D = tsdist_ccf(X, max_lag=2)
        assert D[0, 1] < 0.1  # Should be very small


class TestDynamicTimeWarping:
    """Test cases for Dynamic Time Warping distance."""
    
    def test_tsdist_dtw_identical_series(self):
        """Test DTW distance with identical time series."""
        x = np.array([1, 2, 3, 2, 1])
        X = np.vstack([x, x])
        D = tsdist_dtw(X)
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))
    
    def test_tsdist_dtw_shifted_series(self):
        """Test DTW distance with shifted time series."""
        x1 = np.array([1, 2, 3, 2, 1])
        x2 = np.array([0, 1, 2, 3, 2, 1, 0])
        X = np.vstack([x1, x2])
        D = tsdist_dtw(X)
        assert D[0, 1] > 0  # Should be positive
        assert D[1, 0] == D[0, 1]  # Should be symmetric


class TestInformationTheoreticDistances:
    """Test cases for information-theoretic distance metrics."""
    
    def test_tsdist_nmi_identical_series(self):
        """Test NMI with identical time series."""
        x = np.random.rand(100)
        assert tsdist_nmi(x, x) == 0.0  # Should be 0 for identical series
    
    def test_tsdist_nmi_independent_series(self):
        """Test NMI with independent time series."""
        np.random.seed(42)
        x = np.random.rand(1000)
        y = np.random.rand(1000)
        nmi = tsdist_nmi(x, y)
        assert 0.0 <= nmi <= 1.0  # Should be in [0, 1] range
    
    def test_tsdist_voi_identical_series(self):
        """Test Variation of Information with identical series."""
        x = np.random.rand(100)
        assert tsdist_voi(x, x) == 0.0  # Should be 0 for identical series
    
    @pytest.mark.skipif("minepy" not in globals(), reason="minepy not installed")
    def test_tsdist_mic(self):
        """Test Maximal Information Coefficient."""
        # Linear relationship
        x = np.linspace(0, 1, 100)
        y = 2 * x + 1
        mic = tsdist_mic(x, y)
        assert 0.0 <= mic <= 1.0
        assert mic > 0.8  # Should be close to 1 for linear relationship


class TestVanRossumDistance:
    """Test cases for Van Rossum distance."""
    
    def test_tsdist_vr_identical_spike_trains(self):
        """Test Van Rossum distance with identical spike trains."""
        t = np.array([0.1, 0.2, 0.3, 0.4])
        assert tsdist_vr(t, t) == 0.0
    
    def test_tsdist_vr_different_spike_trains(self):
        """Test Van Rossum distance with different spike trains."""
        t1 = np.array([0.1, 0.2, 0.3])
        t2 = np.array([0.4, 0.5, 0.6])
        d = tsdist_vr(t1, t2, tau=0.1)
        assert d > 0.0
    
    def test_tsdist_vr_empty_spike_train(self):
        """Test Van Rossum distance with one empty spike train."""
        t1 = np.array([0.1, 0.2, 0.3])
        t2 = np.array([])
        d = tsdist_vr(t1, t2, tau=0.1)
        assert d > 0.0


class TestDistanceMatrixUtils:
    """Test cases for distance matrix utility functions."""
    
    def test_dist_percentile(self):
        """Test distance matrix percentile calculation."""
        D = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        assert dist_percentile(D, 0) == 1.0
        assert dist_percentile(D, 50) == 2.0
        assert dist_percentile(D, 100) == 3.0
    
    def test_dist_matrix_normalize_minmax(self):
        """Test min-max normalization of distance matrix."""
        D = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        D_norm = dist_matrix_normalize(D, kind="minmax")
        assert np.allclose(D_norm, np.array([
            [0, 1/3, 2/3],
            [1/3, 0, 1],
            [2/3, 1, 0]
        ]))
    
    def test_dist_matrix_normalize_zscore(self):
        """Test z-score normalization of distance matrix."""
        D = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        D_norm = dist_matrix_normalize(D, kind="zscore")
        mean = np.mean([1, 2, 1, 3, 2, 3])  # Upper triangle values
        std = np.std([1, 2, 1, 3, 2, 3])
        expected = (D - mean) / std if std > 0 else D - mean
        assert np.allclose(D_norm, expected)
    
    def test_dist_matrix_normalize_invalid_kind(self):
        """Test with invalid normalization kind."""
        D = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError):
            dist_matrix_normalize(D, kind="invalid")
