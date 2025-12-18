"""Unit tests for the command-line interface."""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
import numpy as np
from click.testing import CliRunner

# Import the CLI module
from ts2net.cli import cli

class TestCLI:
    """Test cases for the command-line interface."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.runner = CliRunner()
        
        # Create a sample time series file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ts_file = os.path.join(self.temp_dir.name, 'test_series.txt')
        np.savetxt(self.ts_file, np.random.randn(10, 5))  # 10 time points, 5 series
        
        # Create a sample edge list file
        self.edge_file = os.path.join(self.temp_dir.name, 'test_edges.txt')
        with open(self.edge_file, 'w') as f:
            f.write('0,1,1.0\n1,2,0.5\n2,0,0.8\n')
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()
    
    def test_cli_help(self):
        """Test that the CLI shows help information."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Show this message and exit.' in result.output
    
    def test_convert_to_parquet(self):
        """Test converting an edge list to Parquet format."""
        output_dir = os.path.join(self.temp_dir.name, 'output')
        result = self.runner.invoke(
            cli,
            [
                'to-parquet',
                '--name', 'test_graph',
                '--output', output_dir,
                self.edge_file
            ]
        )
        assert result.exit_code == 0
        
        # Check that output files were created
        assert os.path.exists(os.path.join(output_dir, 'nodes.parquet'))
        assert os.path.exists(os.path.join(output_dir, 'edges.parquet'))
        assert os.path.exists(os.path.join(output_dir, 'graph_meta.json'))
    
    def test_convert_to_parquet_directed(self):
        """Test converting a directed graph to Parquet format."""
        output_dir = os.path.join(self.temp_dir.name, 'output_directed')
        result = self.runner.invoke(
            cli,
            [
                'to-parquet',
                '--directed',
                '--name', 'test_directed_graph',
                '--output', output_dir,
                self.edge_file
            ]
        )
        assert result.exit_code == 0
        
        # Check that output files were created
        assert os.path.exists(os.path.join(output_dir, 'nodes.parquet'))
        assert os.path.exists(os.path.join(output_dir, 'edges.parquet'))
    
    @pytest.mark.skipif(
        sys.platform.startswith('win'), 
        reason='GraphML/Graphviz not consistently available on Windows CI'
    )
    def test_convert_from_parquet(self):
        """Test converting from Parquet to other formats (if Graphviz is available)."""
        # First create a Parquet file
        output_dir = os.path.join(self.temp_dir.name, 'output')
        self.runner.invoke(
            cli,
            [
                'to-parquet',
                '--name', 'test_graph',
                '--output', output_dir,
                self.edge_file
            ]
        )
        
        # Test conversion to GraphML
        graphml_file = os.path.join(self.temp_dir.name, 'test.graphml')
        result = self.runner.invoke(
            cli,
            [
                'from-parquet',
                '--graphml', graphml_file,
                os.path.join(output_dir, 'graph_meta.json')
            ]
        )
        
        # On Windows, GraphML might not be available, so we just check for graceful failure
        if 'graphml' in sys.modules or not sys.platform.startswith('win'):
            assert result.exit_code == 0
            assert os.path.exists(graphml_file)
    
    def test_convert_unsupported_format(self):
        """Test error handling for unsupported output formats."""
        # First create a Parquet file
        output_dir = os.path.join(self.temp_dir.name, 'output')
        self.runner.invoke(
            cli,
            [
                'to-parquet',
                '--name', 'test_graph',
                '--output', output_dir,
                self.edge_file
            ]
        )
        
        # Try to convert to an unsupported format
        result = self.runner.invoke(
            cli,
            [
                'from-parquet',
                '--unsupported-format', 'dummy.unsupported',
                os.path.join(output_dir, 'graph_meta.json')
            ]
        )
        
        # Should fail with a non-zero exit code
        assert result.exit_code != 0
        assert ('No such command' in result.output) or ('Error' in result.output) or ('Usage:' in result.output)
    
    @patch('ts2net.cli.commands.spatial.radius_weights')
    def test_spatial_weights_radius(self, mock_radius_weights):
        """Test the spatial weights radius command."""
        # Mock the radius_weights function
        mock_radius_weights.return_value = np.array([[0, 1], [1, 0]])
        
        # Create a temporary coordinates file
        coords_file = os.path.join(self.temp_dir.name, 'coords.txt')
        np.savetxt(coords_file, np.random.rand(5, 2))  # 5 points in 2D
        
        # Run the command
        output_file = os.path.join(self.temp_dir.name, 'weights.txt')
        result = self.runner.invoke(
            cli,
            [
                'spatial-weights',
                'radius',
                coords_file,
                '--radius', '1.0',
                '--output', output_file,
            ]
        )
        
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        mock_radius_weights.assert_called_once()
    
    @patch('ts2net.cli.commands.spatial.knn_weights')
    def test_spatial_weights_knn(self, mock_knn_weights):
        """Test the spatial weights KNN command."""
        # Mock the knn_weights function
        mock_knn_weights.return_value = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        
        # Create a temporary coordinates file
        coords_file = os.path.join(self.temp_dir.name, 'coords_knn.txt')
        np.savetxt(coords_file, np.random.rand(3, 2))  # 3 points in 2D
        
        # Run the command
        output_file = os.path.join(self.temp_dir.name, 'weights_knn.txt')
        result = self.runner.invoke(
            cli,
            [
                'spatial-weights',
                'knn',
                coords_file,
                '--k', '1',
                '--output', output_file,
            ]
        )
        
        assert result.exit_code == 0
        assert os.path.exists(output_file)
        mock_knn_weights.assert_called_once()
    
    def test_spatial_weights_invalid_method(self):
        """Test error handling for invalid spatial weights method."""
        # Create a temporary coordinates file
        coords_file = os.path.join(self.temp_dir.name, 'coords_invalid.txt')
        np.savetxt(coords_file, np.random.rand(3, 2))
        
        # Run with invalid method
        result = self.runner.invoke(
            cli,
            [
                'spatial-weights',
                'invalid_method',
                coords_file,
            ]
        )
        
        assert result.exit_code != 0
        assert 'Error' in result.output
