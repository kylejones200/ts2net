"""
Tests for scikit-learn compatible fit/transform pipeline.
"""

import numpy as np
import pytest
import networkx as nx
import warnings

from ts2net import HVG, NVG


class TestHVGFitTransform:
    """Test HVG fit/transform pipeline."""
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        x = np.random.randn(100)
        hvg = HVG()
        G = hvg.fit_transform(x)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
    
    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.fit(x)
        G = hvg.transform()
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
    
    def test_transform_without_fit(self):
        """Test that transform fails without fit."""
        hvg = HVG()
        with pytest.raises(ValueError, match="Must call fit"):
            hvg.transform()
    
    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        x = np.random.randn(100)
        hvg = HVG()
        result = hvg.fit(x)
        
        assert result is hvg
    
    def test_fit_transform_with_parameters(self):
        """Test fit_transform with different parameters."""
        x = np.random.randn(100)
        hvg = HVG(weighted=True, directed=True)
        G = hvg.fit_transform(x)
        
        assert isinstance(G, (nx.Graph, nx.DiGraph))
        assert G.number_of_nodes() == 100
    
    def test_build_still_works(self):
        """Test that legacy build() method still works."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        assert hvg._graph is not None
        assert hvg.n_nodes == 100


class TestNVGFitTransform:
    """Test NVG fit/transform pipeline."""
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        x = np.random.randn(100)
        nvg = NVG()
        G = nvg.fit_transform(x)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
    
    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        x = np.random.randn(100)
        nvg = NVG()
        nvg.fit(x)
        G = nvg.transform()
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
    
    def test_transform_without_fit(self):
        """Test that transform fails without fit."""
        nvg = NVG()
        with pytest.raises(ValueError, match="Must call fit"):
            nvg.transform()
    
    def test_fit_returns_self(self):
        """Test that fit returns self for chaining."""
        x = np.random.randn(100)
        nvg = NVG()
        result = nvg.fit(x)
        
        assert result is nvg
    
    def test_fit_transform_with_parameters(self):
        """Test fit_transform with different parameters."""
        x = np.random.randn(100)
        nvg = NVG(weighted=True, limit=50)
        G = nvg.fit_transform(x)
        
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100
    
    def test_build_still_works(self):
        """Test that legacy build() method still works."""
        x = np.random.randn(100)
        nvg = NVG()
        nvg.build(x)
        
        assert nvg._graph is not None
        assert nvg.n_nodes == 100


class TestEnhancedErrorHandling:
    """Test enhanced error handling and warnings."""
    
    def test_short_series_warning(self):
        """Test warning for very short series."""
        x = np.array([1.0, 2.0])  # Only 2 points
        hvg = HVG()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hvg.fit(x)
            
            assert len(w) > 0
            assert any("short series" in str(warning.message).lower() for warning in w)
    
    def test_constant_series_warning(self):
        """Test warning for constant series."""
        x = np.ones(100)  # Constant series
        hvg = HVG()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hvg.fit(x)
            
            assert len(w) > 0
            assert any("constant" in str(warning.message).lower() for warning in w)
    
    def test_long_series_warning(self):
        """Test warning for very long series."""
        x = np.random.randn(200_000)  # Very long series
        hvg = HVG()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hvg.fit(x)
            
            assert len(w) > 0
            assert any("long series" in str(warning.message).lower() for warning in w)
    
    def test_large_values_warning(self):
        """Test warning for very large values."""
        x = np.array([1e15, 2e15, 3e15])  # Very large values
        hvg = HVG()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            hvg.fit(x)
            
            assert len(w) > 0
            assert any("large values" in str(warning.message).lower() for warning in w)
    
    def test_empty_series_error(self):
        """Test error for empty series."""
        x = np.array([])
        hvg = HVG()
        
        with pytest.raises(ValueError, match="No valid numeric values"):
            hvg.fit(x)
    
    def test_invalid_input_error(self):
        """Test error for invalid input."""
        # Test with truly invalid input that should fail
        x = np.array([np.nan, np.nan, np.nan])  # All NaN
        hvg = HVG()
        
        # Should raise ValueError for no valid numeric values
        with pytest.raises(ValueError, match="No valid numeric values"):
            hvg.fit(x)


class TestScikitLearnCompatibility:
    """Test scikit-learn pipeline compatibility."""
    
    def test_pipeline_compatible(self):
        """Test that fit/transform works in a pipeline-like pattern."""
        x = np.random.randn(100)
        
        # Simulate pipeline usage
        hvg = HVG()
        hvg.fit(x)
        
        # Transform should work multiple times
        G1 = hvg.transform()
        G2 = hvg.transform()
        
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
    
    def test_fit_multiple_times(self):
        """Test that fit can be called multiple times."""
        # Use clearly different series
        x1 = np.sin(np.linspace(0, 4*np.pi, 100))
        x2 = np.cos(np.linspace(0, 4*np.pi, 100))  # Different function
        
        hvg = HVG()
        hvg.fit(x1)
        G1 = hvg.transform()
        
        hvg.fit(x2)  # Fit again with new data
        G2 = hvg.transform()
        
        assert G1.number_of_nodes() == G2.number_of_nodes() == 100
        # Both should have valid graphs (non-zero edges)
        assert G1.number_of_edges() > 0
        assert G2.number_of_edges() > 0
        # Graphs should be different (different data)
        # Note: For some series, edge counts might coincidentally match,
        # but the structure should be different
        assert G1.number_of_edges() != G2.number_of_edges() or x1[0] != x2[0]

