"""
Tests for neural network-based network inference.

Note: These tests require PyTorch to be installed.
"""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestDynamicsModels:
    """Test dynamics models."""
    
    def test_kuramoto_first_order(self):
        """Test first-order Kuramoto model."""
        from ts2net.inference.neural_inference import KuramotoModel
        
        model = KuramotoModel(order=1, kappa=1.0)
        n_nodes = 5
        A = np.random.rand(n_nodes, n_nodes) * 0.1
        A = (A + A.T) / 2  # Symmetric
        np.fill_diagonal(A, 0)
        
        initial = np.random.rand(n_nodes) * 2 * np.pi
        trajectory = model.forward(A, initial, n_steps=10, dt=0.01)
        
        assert trajectory.shape == (10, n_nodes)
        assert np.all(np.isfinite(trajectory))
    
    def test_kuramoto_second_order(self):
        """Test second-order Kuramoto model."""
        from ts2net.inference.neural_inference import KuramotoModel
        
        model = KuramotoModel(order=2, alpha=1.0, beta=0.2, kappa=1.0)
        n_nodes = 5
        A = np.random.rand(n_nodes, n_nodes) * 0.1
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        
        initial = np.random.rand(n_nodes) * 2 * np.pi
        trajectory = model.forward(A, initial, n_steps=10, dt=0.01)
        
        assert trajectory.shape == (10, n_nodes)
        assert np.all(np.isfinite(trajectory))
    
    def test_linear_dynamics(self):
        """Test linear dynamics model."""
        from ts2net.inference.neural_inference import LinearDynamicsModel
        
        model = LinearDynamicsModel()
        n_nodes = 5
        A = np.random.rand(n_nodes, n_nodes) * 0.1
        np.fill_diagonal(A, 0)
        
        initial = np.random.rand(n_nodes)
        trajectory = model.forward(A, initial, n_steps=10, dt=0.01)
        
        assert trajectory.shape == (10, n_nodes)
        assert np.all(np.isfinite(trajectory))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestAdjacencyNetwork:
    """Test adjacency network architecture."""
    
    def test_network_forward(self):
        """Test network forward pass."""
        from ts2net.inference.neural_inference import AdjacencyNetwork
        
        n_nodes = 5
        input_length = 10
        
        network = AdjacencyNetwork(
            n_nodes=n_nodes,
            input_length=input_length,
            hidden_layers=[10, 10],
        )
        
        x = torch.randn(2, input_length)  # Batch of 2
        A = network(x)
        
        assert A.shape == (2, n_nodes, n_nodes)
        # Check diagonal is zero
        for i in range(n_nodes):
            assert torch.allclose(A[:, i, i], torch.zeros(2))
    
    def test_network_symmetry(self):
        """Test that network enforces symmetry."""
        from ts2net.inference.neural_inference import AdjacencyNetwork
        
        network = AdjacencyNetwork(
            n_nodes=5,
            input_length=10,
            enforce_symmetry=True,
        )
        
        x = torch.randn(1, 10)
        A = network(x)
        
        # Check symmetry
        assert torch.allclose(A[0], A[0].T)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestNeuralNetworkInference:
    """Test neural network inference."""
    
    def test_initialization(self):
        """Test initialization."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        dynamics = LinearDynamicsModel()
        infer = NeuralNetworkInference(
            n_nodes=5,
            dynamics_model=dynamics,
            hidden_layers=[10, 10],
        )
        
        assert infer.n_nodes == 5
        assert infer.network is None  # Not created until fit
    
    def test_fit_predict_linear(self):
        """Test fitting and prediction with linear dynamics."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        np.random.seed(42)
        n_nodes = 5
        n_steps = 20
        
        # Create synthetic data
        A_true = np.random.rand(n_nodes, n_nodes) * 0.5
        A_true = (A_true + A_true.T) / 2  # Symmetric
        np.fill_diagonal(A_true, 0)
        
        dynamics = LinearDynamicsModel()
        initial = np.random.rand(n_nodes)
        X_true = dynamics.forward(A_true, initial, n_steps=n_steps, dt=0.01)
        
        # Add small noise
        X_obs = X_true + np.random.randn(*X_true.shape) * 0.01
        
        # Fit model
        infer = NeuralNetworkInference(
            n_nodes=n_nodes,
            dynamics_model=dynamics,
            hidden_layers=[10, 10],
            learning_rate=1e-2,
        )
        
        infer.fit(X_obs, initial_conditions=initial, n_epochs=50, verbose=False)
        
        # Predict
        A_pred = infer.predict(X_obs)
        
        assert A_pred.shape == (n_nodes, n_nodes)
        assert np.all(np.isfinite(A_pred))
        # Diagonal should be zero
        assert np.allclose(np.diag(A_pred), 0, atol=1e-3)
    
    def test_uncertainty_quantification(self):
        """Test uncertainty quantification."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        np.random.seed(42)
        n_nodes = 5
        n_steps = 20
        
        dynamics = LinearDynamicsModel()
        initial = np.random.rand(n_nodes)
        A_true = np.random.rand(n_nodes, n_nodes) * 0.5
        A_true = (A_true + A_true.T) / 2
        np.fill_diagonal(A_true, 0)
        
        X_true = dynamics.forward(A_true, initial, n_steps=n_steps, dt=0.01)
        X_obs = X_true + np.random.randn(*X_true.shape) * 0.01
        
        infer = NeuralNetworkInference(
            n_nodes=n_nodes,
            dynamics_model=dynamics,
            hidden_layers=[10, 10],
        )
        
        infer.fit(X_obs, initial_conditions=initial, n_epochs=30, verbose=False)
        
        uncertainty = infer.get_uncertainty()
        
        assert 'mean' in uncertainty
        assert 'std' in uncertainty
        assert 'mode' in uncertainty
        assert uncertainty['mean'].shape == (n_nodes, n_nodes)
        assert uncertainty['std'].shape == (n_nodes, n_nodes)
        assert uncertainty['mode'].shape == (n_nodes, n_nodes)
    
    def test_loss_history(self):
        """Test that loss history is tracked."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        np.random.seed(42)
        dynamics = LinearDynamicsModel()
        initial = np.random.rand(5)
        A_true = np.random.rand(5, 5) * 0.5
        A_true = (A_true + A_true.T) / 2
        np.fill_diagonal(A_true, 0)
        
        X_true = dynamics.forward(A_true, initial, n_steps=20, dt=0.01)
        
        infer = NeuralNetworkInference(
            n_nodes=5,
            dynamics_model=dynamics,
        )
        
        infer.fit(X_true, initial_conditions=initial, n_epochs=20, verbose=False)
        
        assert len(infer.loss_history) == 20
        assert len(infer.adjacency_history) == 20
        # Loss should generally decrease (not strictly, but usually)
        assert infer.loss_history[-1] < infer.loss_history[0] * 2  # Allow some variance
    
    def test_predict_before_fit_error(self):
        """Test error when predicting before fitting."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        dynamics = LinearDynamicsModel()
        infer = NeuralNetworkInference(
            n_nodes=5,
            dynamics_model=dynamics,
        )
        
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="must be fitted"):
            infer.predict(X)
    
    def test_uncertainty_before_fit_error(self):
        """Test error when getting uncertainty before fitting."""
        from ts2net.inference.neural_inference import NeuralNetworkInference, LinearDynamicsModel
        
        dynamics = LinearDynamicsModel()
        infer = NeuralNetworkInference(
            n_nodes=5,
            dynamics_model=dynamics,
        )
        
        with pytest.raises(ValueError, match="No training history"):
            infer.get_uncertainty()


@pytest.mark.skipif(TORCH_AVAILABLE, reason="PyTorch is installed")
def test_import_error():
    """Test that ImportError is raised when PyTorch is not available."""
    try:
        from ts2net.inference.neural_inference import NeuralNetworkInference
        pytest.skip("PyTorch is available, cannot test ImportError")
    except ImportError as e:
        assert "PyTorch is required" in str(e)

