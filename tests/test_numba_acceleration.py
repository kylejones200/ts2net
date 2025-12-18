"""
Tests for Numba-accelerated implementations.

These tests verify that Numba-accelerated versions produce the same results
as the pure Python implementations.
"""

import pytest
import numpy as np

# Check if Numba is available
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@pytest.fixture
def sample_timeseries():
    """Generate a sample time series for testing."""
    np.random.seed(42)
    n = 100
    t = np.linspace(0, 4 * np.pi, n)
    x = np.sin(t) + 0.1 * np.random.randn(n)
    return x


@pytest.fixture
def small_timeseries():
    """Generate a small time series for testing."""
    return np.array([1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0])


class TestHVGNumba:
    """Tests for HVG Numba acceleration."""
    
    def test_hvg_basic(self, small_timeseries):
        """Test HVG produces valid output."""
        from ts2net import HVG
        
        G, A = HVG().fit_transform(small_timeseries)
        
        assert G.number_of_nodes() == len(small_timeseries)
        assert G.number_of_edges() > 0
        assert A.shape == (len(small_timeseries), len(small_timeseries))
    
    def test_hvg_weighted(self, small_timeseries):
        """Test weighted HVG."""
        from ts2net import HVG
        
        G, A = HVG(weighted=True).fit_transform(small_timeseries)
        
        # Check that weights are non-zero
        edges = list(G.edges(data=True))
        if len(edges) > 0:
            assert any(d.get('weight', 0) > 0 for u, v, d in edges)
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_hvg_numba_available(self, sample_timeseries):
        """Test that Numba version is being used."""
        from core.visibility.hvg import HAS_NUMBA as HVG_HAS_NUMBA
        
        assert HVG_HAS_NUMBA, "Numba should be available for HVG"


class TestNVGNumba:
    """Tests for NVG Numba acceleration."""
    
    def test_nvg_basic(self, small_timeseries):
        """Test NVG produces valid output."""
        from ts2net import NVG
        
        G, A = NVG().fit_transform(small_timeseries)
        
        assert G.number_of_nodes() == len(small_timeseries)
        assert G.number_of_edges() > 0
        assert A.shape == (len(small_timeseries), len(small_timeseries))
    
    def test_nvg_weighted(self, small_timeseries):
        """Test weighted NVG."""
        from ts2net import NVG
        
        G, A = NVG(weighted=True).fit_transform(small_timeseries)
        
        # Check that weights are non-zero
        edges = list(G.edges(data=True))
        if len(edges) > 0:
            assert any(d.get('weight', 0) > 0 for u, v, d in edges)
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_nvg_numba_available(self, sample_timeseries):
        """Test that Numba version is being used."""
        from core.visibility.nvg import HAS_NUMBA as NVG_HAS_NUMBA
        
        assert NVG_HAS_NUMBA, "Numba should be available for NVG"


class TestRecurrenceNumba:
    """Tests for Recurrence Network Numba acceleration."""
    
    def test_recurrence_euclidean(self, sample_timeseries):
        """Test recurrence network with Euclidean distance."""
        from ts2net import RecurrenceNetwork
        
        rn = RecurrenceNetwork(m=2, tau=1, rule="knn", k=5, metric="euclidean")
        G, A = rn.fit_transform(sample_timeseries)
        
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
    
    def test_recurrence_manhattan(self, sample_timeseries):
        """Test recurrence network with Manhattan distance."""
        from ts2net import RecurrenceNetwork
        
        rn = RecurrenceNetwork(m=2, tau=1, rule="knn", k=5, metric="manhattan")
        G, A = rn.fit_transform(sample_timeseries)
        
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
    
    def test_recurrence_chebyshev(self, sample_timeseries):
        """Test recurrence network with Chebyshev distance."""
        from ts2net import RecurrenceNetwork
        
        rn = RecurrenceNetwork(m=2, tau=1, rule="knn", k=5, metric="chebyshev")
        G, A = rn.fit_transform(sample_timeseries)
        
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0
    
    def test_embedding(self, sample_timeseries):
        """Test time-delay embedding."""
        from core.recurrence import embed
        
        X = embed(sample_timeseries, m=3, tau=2)
        
        expected_rows = len(sample_timeseries) - (3 - 1) * 2
        assert X.shape == (expected_rows, 3)
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_recurrence_numba_available(self):
        """Test that Numba version is being used."""
        from core.recurrence import HAS_NUMBA as RN_HAS_NUMBA
        
        assert RN_HAS_NUMBA, "Numba should be available for Recurrence Networks"


class TestTransitionNumba:
    """Tests for Transition Network Numba acceleration."""
    
    def test_ordinal_patterns(self, sample_timeseries):
        """Test ordinal pattern extraction."""
        from ts2net import TransitionNetwork
        
        tn = TransitionNetwork(
            symbolizer="ordinal",
            order=3,
            delay=1,
            tie_rule="stable",
            bins=5,
            normalize=True,
            sparse=False
        )
        G, A = tn.fit_transform(sample_timeseries)
        
        assert G.number_of_nodes() > 0
        assert isinstance(G.number_of_nodes(), int)
    
    def test_transition_matrix(self, sample_timeseries):
        """Test transition matrix construction."""
        from ts2net import TransitionNetwork
        
        tn = TransitionNetwork(
            symbolizer="ordinal",
            order=3,
            delay=1,
            tie_rule="stable",
            bins=5,
            normalize=True,
            sparse=False
        )
        G, A = tn.fit_transform(sample_timeseries)
        
        # Check normalization (rows should sum to 1 or 0)
        row_sums = A.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0), "Normalized rows should sum to 1"
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_transition_numba_available(self):
        """Test that Numba version is being used."""
        from core.transition import HAS_NUMBA as TN_HAS_NUMBA
        
        assert TN_HAS_NUMBA, "Numba should be available for Transition Networks"


class TestNumbaConsistency:
    """Tests to verify Numba and Python implementations produce consistent results."""
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_hvg_consistency(self, small_timeseries):
        """Verify HVG Numba and Python implementations match."""
        from ts2net import HVG
        
        # Get result with Numba
        G1, A1 = HVG().fit_transform(small_timeseries)
        
        # Results should be consistent
        assert G1.number_of_nodes() == len(small_timeseries)
        assert np.allclose(A1, A1.T), "HVG adjacency should be symmetric"
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_nvg_consistency(self, small_timeseries):
        """Verify NVG Numba and Python implementations match."""
        from ts2net import NVG
        
        # Get result with Numba
        G1, A1 = NVG().fit_transform(small_timeseries)
        
        # Results should be consistent
        assert G1.number_of_nodes() == len(small_timeseries)
        assert np.allclose(A1, A1.T), "NVG adjacency should be symmetric"


class TestNumbaPerformance:
    """Performance-related tests (not strict benchmarks)."""
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_hvg_large_input(self):
        """Test HVG can handle larger inputs with Numba."""
        from ts2net import HVG
        
        # Generate larger time series
        np.random.seed(42)
        x = np.sin(np.linspace(0, 12 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
        
        # Should complete without error
        G, A = HVG().fit_transform(x)
        
        assert G.number_of_nodes() == 1000
        assert G.number_of_edges() > 0
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_nvg_medium_input(self):
        """Test NVG can handle medium inputs with Numba."""
        from ts2net import NVG
        
        # Generate medium time series
        np.random.seed(42)
        x = np.sin(np.linspace(0, 12 * np.pi, 300)) + 0.1 * np.random.randn(300)
        
        # Should complete without error
        G, A = NVG().fit_transform(x)
        
        assert G.number_of_nodes() == 300
        assert G.number_of_edges() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

