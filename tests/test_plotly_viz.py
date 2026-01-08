"""
Tests for Plotly-based interactive visualizations.

Note: These tests require plotly to be installed.
"""

import numpy as np
import pytest
import networkx as nx

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPlotlyVisualizations:
    """Test Plotly-based interactive visualizations."""
    
    def test_plot_timeseries_network_basic(self):
        """Test basic time series network plotting."""
        from ts2net.viz.plotly_viz import plot_timeseries_network
        
        # Create simple graphs
        graphs = [
            nx.erdos_renyi_graph(10, 0.3, seed=42),
            nx.erdos_renyi_graph(10, 0.4, seed=43),
            nx.erdos_renyi_graph(10, 0.5, seed=44),
        ]
        timestamps = ["Step 1", "Step 2", "Step 3"]
        
        fig = plot_timeseries_network(
            graphs=graphs,
            timestamps=timestamps,
            show=False,
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6  # 3 time steps * 2 traces (edges + nodes)
        assert len(fig.layout.sliders) == 1
        assert len(fig.layout.sliders[0].steps) == 3
    
    def test_plot_timeseries_network_with_pos(self):
        """Test with custom node positions."""
        from ts2net.viz.plotly_viz import plot_timeseries_network
        
        graphs = [nx.erdos_renyi_graph(5, 0.5, seed=42) for _ in range(2)]
        timestamps = ["T1", "T2"]
        
        # Custom positions
        pos = {i: np.array([i, i*0.5]) for i in range(5)}
        
        fig = plot_timeseries_network(
            graphs=graphs,
            timestamps=timestamps,
            pos=pos,
            show=False,
        )
        
        assert isinstance(fig, go.Figure)
    
    def test_plot_timeseries_network_node_colors(self):
        """Test different node coloring schemes."""
        from ts2net.viz.plotly_viz import plot_timeseries_network
        
        graphs = [nx.erdos_renyi_graph(5, 0.5, seed=42)]
        timestamps = ["Step 1"]
        
        # Test degree coloring
        fig1 = plot_timeseries_network(
            graphs=graphs,
            timestamps=timestamps,
            node_colors="degree",
            show=False,
        )
        assert isinstance(fig1, go.Figure)
        
        # Test time coloring
        fig2 = plot_timeseries_network(
            graphs=graphs,
            timestamps=timestamps,
            node_colors="time",
            show=False,
        )
        assert isinstance(fig2, go.Figure)
    
    def test_plot_windowed_networks(self):
        """Test windowed network visualization."""
        from ts2net.viz.plotly_viz import plot_windowed_networks
        
        np.random.seed(42)
        x = np.random.randn(200)
        
        fig = plot_windowed_networks(
            x=x,
            window=50,
            step=25,
            method='hvg',
            show=False,
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.sliders) == 1
    
    def test_plot_windowed_networks_invalid_method(self):
        """Test error handling for invalid method."""
        from ts2net.viz.plotly_viz import plot_windowed_networks
        
        x = np.random.randn(100)
        
        with pytest.raises(ValueError, match="Unknown method"):
            plot_windowed_networks(x, window=20, method='invalid', show=False)
    
    def test_plot_timeseries_network_mismatched_lengths(self):
        """Test error for mismatched graph and timestamp lengths."""
        from ts2net.viz.plotly_viz import plot_timeseries_network
        
        graphs = [nx.erdos_renyi_graph(5, 0.5, seed=42)]
        timestamps = ["T1", "T2"]  # Mismatch
        
        with pytest.raises(ValueError, match="must match"):
            plot_timeseries_network(graphs, timestamps, show=False)
    
    def test_plot_timeseries_network_empty(self):
        """Test error for empty graph list."""
        from ts2net.viz.plotly_viz import plot_timeseries_network
        
        with pytest.raises(ValueError, match="At least one graph"):
            plot_timeseries_network([], [], show=False)


@pytest.mark.skipif(PLOTLY_AVAILABLE, reason="Plotly is installed")
def test_plotly_import_error():
    """Test that ImportError is raised when plotly is not available."""
    # This test only runs when plotly is NOT installed
    try:
        from ts2net.viz.plotly_viz import plot_timeseries_network
        # If we get here, plotly is available, so we can't test the error
        pytest.skip("Plotly is available, cannot test ImportError")
    except ImportError as e:
        assert "Plotly is required" in str(e)

