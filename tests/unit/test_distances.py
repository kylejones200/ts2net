"""Unit tests for distance metrics — correctness, not just shapes."""
import numpy as np
import pytest
from ts2net.distances.core import (
    tsdist_cor,
    tsdist_ccf,
    tsdist_dtw,
    tsdist_nmi,
    tsdist_voi,
    tsdist_mic,
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
        # Self-distance should be 0 (allowing for floating point precision)
        assert np.isclose(D[0, 0], 0.0, atol=1e-6)


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
    """Tests for information-theoretic distances — correctness, not just shapes."""

    def test_tsdist_nmi_identical_series(self):
        """NMI of a series with itself is 0 (perfect dependence → distance 0)."""
        np.random.seed(42)
        x = np.random.rand(100)
        assert np.isclose(tsdist_nmi(x, x), 0.0, atol=1e-10)

    def test_tsdist_voi_identical_series(self):
        """VOI of a series with itself is 0."""
        np.random.seed(42)
        x = np.random.rand(100)
        assert np.isclose(tsdist_voi(x, x), 0.0, atol=1e-10)

    def test_tsdist_voi_always_non_negative(self):
        """
        VOI must be >= 0 for any pair of series.

        Regression test: the previous formula (hx + hy - 2*hxy) was the
        negation of the correct formula (2*hxy - hx - hy) and routinely
        returned large negative values for independent series.
        """
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.standard_normal(200)
            y = rng.standard_normal(200)
            v = tsdist_voi(x, y)
            assert v >= 0, f"VOI must be >= 0, got {v}"

    def test_tsdist_voi_symmetric(self):
        """VOI(x, y) == VOI(y, x)."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        assert np.isclose(tsdist_voi(x, y), tsdist_voi(y, x), atol=1e-10)

    def test_tsdist_voi_independent_greater_than_zero(self):
        """Independent series should have VOI > 0."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        assert tsdist_voi(x, y) > 0.0

    def test_tsdist_mic_raises_without_minepy(self):
        """
        tsdist_mic must raise ImportError (not silently return NMI).

        MIC and NMI are different metrics. Silently substituting one for
        the other produces incorrect results without any indication.
        """
        import importlib
        import sys
        # Force minepy to appear absent for this test
        original = sys.modules.get("minepy")
        sys.modules["minepy"] = None  # type: ignore[assignment]
        try:
            import importlib.util
            import ts2net.distances.core as dc
            orig_minepy = dc.minepy
            dc.minepy = None
            with pytest.raises(ImportError, match="minepy"):
                tsdist_mic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        finally:
            dc.minepy = orig_minepy
            if original is None:
                sys.modules.pop("minepy", None)
            else:
                sys.modules["minepy"] = original


class TestDistanceMatrixUtils:
    """Basic tests for distance matrix utilities."""
    
    def test_dist_matrix_normalize_minmax(self):
        """Test min-max normalization."""
        D = np.array([[0, 5, 10], [5, 0, 8], [10, 8, 0]])
        D_norm = dist_matrix_normalize(D, kind="minmax")
        assert D_norm.shape == D.shape
        assert np.all(D_norm >= 0)
        assert np.all(D_norm <= 1)
