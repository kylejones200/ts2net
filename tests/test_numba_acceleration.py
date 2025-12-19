"""
Basic tests for Numba-accelerated implementations.
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
def small_timeseries():
    """Generate a small time series for testing."""
    return np.array([1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0])


class TestHVGNumba:
    """Basic tests for HVG Numba acceleration."""
    
    def test_hvg_basic(self, small_timeseries):
        """Test HVG produces valid output."""
        from ts2net import HVG
        
        hvg = HVG()
        hvg.build(small_timeseries)
        G = hvg.as_networkx()
        A = hvg.adjacency_matrix()
        
        assert G.number_of_nodes() == len(small_timeseries)
        assert G.number_of_edges() > 0
        assert A.shape == (len(small_timeseries), len(small_timeseries))
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_hvg_numba_available(self):
        """Test that Numba version is being used."""
        from ts2net.core.visibility.hvg import HAS_NUMBA as HVG_HAS_NUMBA
        assert HVG_HAS_NUMBA, "Numba should be available for HVG"


class TestNVGNumba:
    """Basic tests for NVG Numba acceleration."""
    
    def test_nvg_basic(self, small_timeseries):
        """Test NVG produces valid output."""
        from ts2net import NVG
        
        nvg = NVG()
        nvg.build(small_timeseries)
        G = nvg.as_networkx()
        A = nvg.adjacency_matrix()
        
        assert G.number_of_nodes() == len(small_timeseries)
        assert G.number_of_edges() > 0
        assert A.shape == (len(small_timeseries), len(small_timeseries))
    
    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_nvg_numba_available(self):
        """Test that Numba version is being used."""
        from ts2net.core.visibility.nvg import HAS_NUMBA as NVG_HAS_NUMBA
        assert NVG_HAS_NUMBA, "Numba should be available for NVG"
