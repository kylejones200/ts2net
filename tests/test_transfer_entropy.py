"""
Tests for transfer entropy and causal inference methods.
"""

import numpy as np
import pytest
import networkx as nx
from ts2net.causal.transfer_entropy import (
    transfer_entropy,
    transfer_entropy_network,
    conditional_transfer_entropy,
)


class TestTransferEntropy:
    """Test transfer entropy computation."""
    
    def test_transfer_entropy_independent(self):
        """Transfer entropy should be near zero for independent series."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        
        te = transfer_entropy(x, y, lag=1)
        
        assert te >= 0, "Transfer entropy must be non-negative"
        assert te < 0.5, "Independent series should have low transfer entropy"
    
    def test_transfer_entropy_causal(self):
        """Transfer entropy should be positive for causal relationships."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = 0.5 * x[:-1] + 0.1 * np.random.randn(999)
        y = np.concatenate([[0], y])
        
        te = transfer_entropy(x, y, lag=1)
        
        assert te > 0, "Causal relationship should have positive transfer entropy"
        assert te < 10, "Transfer entropy should be reasonable (not too large)"
    
    def test_transfer_entropy_symmetric(self):
        """Transfer entropy is asymmetric (TE(X→Y) != TE(Y→X) in general)."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = 0.5 * x[:-1] + 0.1 * np.random.randn(999)
        y = np.concatenate([[0], y])
        
        te_xy = transfer_entropy(x, y, lag=1)
        te_yx = transfer_entropy(y, x, lag=1)
        
        assert te_xy != te_yx, "Transfer entropy should be asymmetric"
        assert te_xy > te_yx, "X→Y should have higher TE than Y→X for this relationship"
    
    def test_transfer_entropy_lag(self):
        """Transfer entropy should vary with lag."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = 0.5 * x[:-2] + 0.1 * np.random.randn(998)
        y = np.concatenate([[0, 0], y])
        
        te_lag1 = transfer_entropy(x, y, lag=1)
        te_lag2 = transfer_entropy(x, y, lag=2)
        
        assert te_lag2 > te_lag1, "Higher lag should capture the relationship better"
    
    def test_transfer_entropy_methods(self):
        """Both discrete and knn methods should work."""
        np.random.seed(42)
        x = np.random.randn(500)
        y = 0.5 * x[:-1] + 0.1 * np.random.randn(499)
        y = np.concatenate([[0], y])
        
        te_discrete = transfer_entropy(x, y, method="discrete")
        te_knn = transfer_entropy(x, y, method="knn")
        
        assert te_discrete >= 0
        assert te_knn >= 0
        assert abs(te_discrete - te_knn) < 2.0, "Methods should give similar results"
    
    def test_transfer_entropy_short_series(self):
        """Should handle short series gracefully."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        
        te = transfer_entropy(x, y, lag=1)
        
        assert te >= 0
        assert not np.isnan(te)
        assert not np.isinf(te)


class TestConditionalTransferEntropy:
    """Test conditional transfer entropy computation."""
    
    def test_conditional_transfer_entropy(self):
        """Conditional transfer entropy should account for confounding."""
        np.random.seed(42)
        x = np.random.randn(1000)
        z = np.random.randn(1000)
        y = 0.3 * x[:-1] + 0.3 * z[:-1] + 0.1 * np.random.randn(999)
        y = np.concatenate([[0], y])
        
        cte = conditional_transfer_entropy(x, y, z, lag=1)
        
        assert cte >= 0, "Conditional transfer entropy must be non-negative"
        assert cte < 5, "Should be reasonable value"
    
    def test_conditional_vs_unconditional(self):
        """Conditional TE should account for confounding."""
        np.random.seed(42)
        x = np.random.randn(1000)
        z = np.random.randn(1000)
        y = 0.3 * x[:-1] + 0.5 * z[:-1] + 0.1 * np.random.randn(999)
        y = np.concatenate([[0], y])
        
        te = transfer_entropy(x, y, lag=1)
        cte = conditional_transfer_entropy(x, y, z, lag=1)
        
        assert cte >= 0, "Conditional TE must be non-negative"
        assert te >= 0, "Unconditional TE must be non-negative"
        assert abs(cte - te) < 5.0, "Conditional and unconditional TE should be similar in magnitude"


class TestTransferEntropyNetwork:
    """Test transfer entropy network construction."""
    
    def test_transfer_entropy_network_basic(self):
        """Test basic network construction."""
        np.random.seed(42)
        X = [np.random.randn(500) for _ in range(3)]
        
        G, te_matrix, stats = transfer_entropy_network(X, lag=1)
        
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert te_matrix.shape == (3, 3)
        assert np.all(np.diag(te_matrix) == 0), "Diagonal should be zero"
        assert 'mean_te' in stats
        assert 'n_edges' in stats
    
    def test_transfer_entropy_network_threshold(self):
        """Test network with threshold filtering."""
        np.random.seed(42)
        X = [np.random.randn(500) for _ in range(3)]
        
        G_no_thresh, _, _ = transfer_entropy_network(X, threshold=None)
        G_thresh, _, _ = transfer_entropy_network(X, threshold=0.1)
        
        assert G_thresh.number_of_edges() <= G_no_thresh.number_of_edges()
    
    def test_transfer_entropy_network_causal_structure(self):
        """Test network captures causal structure."""
        np.random.seed(42)
        x1 = np.random.randn(1000)
        x2 = 0.5 * x1[:-1] + 0.1 * np.random.randn(999)
        x2 = np.concatenate([[0], x2])
        x3 = np.random.randn(1000)
        
        X = [x1, x2, x3]
        G, te_matrix, _ = transfer_entropy_network(X, lag=1)
        
        assert G.has_edge(0, 1), "Should have edge from x1 to x2"
        assert te_matrix[0, 1] > te_matrix[0, 2], "x1→x2 should have higher TE than x1→x3"
    
    def test_transfer_entropy_network_series_names(self):
        """Test network with custom series names."""
        np.random.seed(42)
        X = [np.random.randn(500) for _ in range(3)]
        names = ["Sensor_A", "Sensor_B", "Sensor_C"]
        
        G, _, _ = transfer_entropy_network(X, series_names=names)
        
        assert G.nodes[0]['name'] == "Sensor_A"
        assert G.nodes[1]['name'] == "Sensor_B"
        assert G.nodes[2]['name'] == "Sensor_C"

