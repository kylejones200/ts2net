"""Simplified unit tests for distance metrics."""
import numpy as np
import pytest
from ts2net.distances.core import (
    tsdist_cor,
    tsdist_ccf,
    tsdist_dtw,
    tsdist_nmi,
    tsdist_voi,
    dist_matrix_normalize,
)


class TestCorrelationDistance:
    """Basic tests for correlation-based distance metrics."""
    
    def test_tsdist_cor_pearson(self):
        """Test Pearson correlation distance."""
        x1 = np.array([1, 2, 3, 4, 5])
        x2 = np.array([2, 4, 6, 8, 10])
        X = np.vstack([x1, x2])
        D = tsdist_cor(X, method="pearson")
        assert D.shape == (2, 2)
        assert np.isclose(D[0, 0], 0.0, atol=1e-10)  # Self-distance should be 0


class TestCCFDistance:
    """Basic tests for cross-correlation based distance."""
    
    def test_tsdist_ccf_identical_series(self):
        """Test CCF distance with identical time series."""
        x = np.array([1, 2, 3, 2, 1])
        X = np.vstack([x, x])
        D = tsdist_ccf(X, max_lag=2)
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))


class TestDynamicTimeWarping:
    """Basic tests for Dynamic Time Warping distance."""
    
    def test_tsdist_dtw_identical_series(self):
        """Test DTW distance with identical time series."""
        x = np.array([1, 2, 3, 2, 1])
        X = np.vstack([x, x])
        D = tsdist_dtw(X)
        assert np.allclose(D, np.array([[0, 0], [0, 0]]))


class TestInformationTheoreticDistances:
    """Basic tests for information-theoretic distance metrics."""
    
    def test_tsdist_nmi_identical_series(self):
        """Test NMI with identical time series."""
        x = np.random.rand(100)
        assert np.isclose(tsdist_nmi(x, x), 0.0, atol=1e-10)
    
    def test_tsdist_voi_identical_series(self):
        """Test VOI with identical time series."""
        x = np.random.rand(100)
        assert np.isclose(tsdist_voi(x, x), 0.0, atol=1e-10)


class TestDistanceMatrixUtils:
    """Basic tests for distance matrix utilities."""
    
    def test_dist_matrix_normalize_minmax(self):
        """Test min-max normalization."""
        D = np.array([[0, 5, 10], [5, 0, 8], [10, 8, 0]])
        D_norm = dist_matrix_normalize(D, kind="minmax")
        assert D_norm.shape == D.shape
        assert np.all(D_norm >= 0)
        assert np.all(D_norm <= 1)
