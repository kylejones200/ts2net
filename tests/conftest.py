"""
Test configuration and fixtures for pytest.

This file contains fixtures and configuration that will be available to all tests.
"""
import numpy as np
import pytest

# Set random seed for reproducibility
np.random.seed(42)

# Common test data fixtures
@pytest.fixture
def sample_time_series():
    """Generate a sample time series for testing."""
    return np.random.randn(100)

@pytest.fixture
def sample_2d_time_series():
    """Generate a 2D array of time series for testing."""
    return np.random.randn(5, 100)  # 5 time series, each with 100 points

@pytest.fixture
def sample_adjacency_matrix():
    """Generate a sample adjacency matrix for testing."""
    n = 10
    adj = np.random.rand(n, n)
    adj = (adj + adj.T) / 2  # Make symmetric
    np.fill_diagonal(adj, 0)  # No self-loops
    return adj

@pytest.fixture
def sample_undirected_graph():
    """Generate a sample undirected graph for testing."""
    import networkx as nx
    return nx.erdos_renyi_graph(10, 0.3)
