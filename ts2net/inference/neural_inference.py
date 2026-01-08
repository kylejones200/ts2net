"""
Neural network-based inference of network adjacency matrices from time series.

Implements the method from Gaskin et al. (2024) for inferring network structures
from observed dynamics using neural networks with uncertainty quantification.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Callable, Literal
import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None


class DynamicsModel:
    """
    Base class for dynamics models used in network inference.
    
    Subclasses should implement the forward method that simulates
    time series given an adjacency matrix.
    """
    
    def forward(
        self,
        A: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        n_steps: int,
        dt: float = 0.01,
        **params
    ) -> NDArray[np.float64]:
        """
        Simulate dynamics given adjacency matrix.
        
        Parameters
        ----------
        A : array (n_nodes, n_nodes)
            Adjacency matrix
        initial_conditions : array (n_nodes,)
            Initial state
        n_steps : int
            Number of time steps to simulate
        dt : float, default 0.01
            Time step size
        **params
            Additional model parameters
        
        Returns
        -------
        array (n_steps, n_nodes)
            Simulated time series
        """
        raise NotImplementedError


class KuramotoModel(DynamicsModel):
    """
    Kuramoto model of coupled oscillators.
    
    Models network of oscillators with phases that synchronize according to:
    dθ_i/dt = ω_i + κ * Σ_j A_ij * sin(θ_j - θ_i)
    
    For second-order dynamics:
    d²θ_i/dt² = -α * dθ_i/dt + ω_i + κ * Σ_j A_ij * sin(θ_j - θ_i)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        kappa: float = 60.0,
        order: int = 1,
    ):
        """
        Initialize Kuramoto model.
        
        Parameters
        ----------
        alpha : float, default 1.0
            Inertia coefficient (for second-order)
        beta : float, default 0.2
            Friction coefficient (for second-order)
        kappa : float, default 60.0
            Coupling strength
        order : int, default 1
            Order of dynamics: 1 (first-order) or 2 (second-order)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.order = order
    
    def forward(
        self,
        A: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        n_steps: int,
        dt: float = 0.01,
        omega: Optional[NDArray[np.float64]] = None,
        **params
    ) -> NDArray[np.float64]:
        """
        Simulate Kuramoto dynamics.
        
        Parameters
        ----------
        A : array (n_nodes, n_nodes)
            Adjacency matrix
        initial_conditions : array (n_nodes,) or (2, n_nodes) for second-order
            Initial phases (and velocities for second-order)
        n_steps : int
            Number of time steps
        dt : float, default 0.01
            Time step size
        omega : array (n_nodes,), optional
            Natural frequencies. If None, uses zeros.
        **params
            Override model parameters (alpha, beta, kappa)
        
        Returns
        -------
        array (n_steps, n_nodes)
            Phase time series
        """
        n_nodes = A.shape[0]
        
        # Get parameters (allow override)
        alpha = params.get('alpha', self.alpha)
        beta = params.get('beta', self.beta)
        kappa = params.get('kappa', self.kappa)
        
        # Natural frequencies
        if omega is None:
            omega = np.zeros(n_nodes)
        else:
            omega = np.asarray(omega, dtype=np.float64)
        
        # Initialize state
        if self.order == 1:
            theta = np.asarray(initial_conditions, dtype=np.float64).copy()
            if theta.ndim == 0:
                theta = theta.reshape(1, -1)
            if theta.shape[0] == 1:
                theta = theta[0]
        else:  # second-order
            if initial_conditions.ndim == 1:
                theta = initial_conditions.copy()
                dtheta = np.zeros_like(theta)
            else:
                theta = initial_conditions[0].copy()
                dtheta = initial_conditions[1].copy()
        
        # Storage
        trajectory = np.zeros((n_steps, n_nodes), dtype=np.float64)
        
        # Time stepping
        for t in range(n_steps):
            trajectory[t] = theta.copy()
            
            if self.order == 1:
                # First-order: dθ/dt = ω + κ * Σ A_ij * sin(θ_j - θ_i)
                coupling = np.zeros(n_nodes)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if A[i, j] > 0:
                            coupling[i] += A[i, j] * np.sin(theta[j] - theta[i])
                
                dtheta_dt = omega + kappa * coupling
                theta += dt * dtheta_dt
            else:
                # Second-order: d²θ/dt² = -α*dθ/dt + ω + κ * Σ A_ij * sin(θ_j - θ_i)
                coupling = np.zeros(n_nodes)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if A[i, j] > 0:
                            coupling[i] += A[i, j] * np.sin(theta[j] - theta[i])
                
                d2theta_dt2 = -alpha * dtheta + omega + kappa * coupling
                dtheta += dt * d2theta_dt2
                theta += dt * dtheta
        
        return trajectory


class LinearDynamicsModel(DynamicsModel):
    """
    Linear dynamics model: dx/dt = -x + A @ x
    
    Simple linear model for testing and comparison.
    """
    
    def forward(
        self,
        A: NDArray[np.float64],
        initial_conditions: NDArray[np.float64],
        n_steps: int,
        dt: float = 0.01,
        **params
    ) -> NDArray[np.float64]:
        """
        Simulate linear dynamics.
        
        Parameters
        ----------
        A : array (n_nodes, n_nodes)
            Adjacency matrix
        initial_conditions : array (n_nodes,)
            Initial state
        n_steps : int
            Number of time steps
        dt : float, default 0.01
            Time step size
        **params
            Additional parameters (not used)
        
        Returns
        -------
        array (n_steps, n_nodes)
            State time series
        """
        x = np.asarray(initial_conditions, dtype=np.float64).copy()
        n_nodes = len(x)
        
        trajectory = np.zeros((n_steps, n_nodes), dtype=np.float64)
        
        for t in range(n_steps):
            trajectory[t] = x.copy()
            # dx/dt = -x + A @ x
            dx_dt = -x + A @ x
            x += dt * dx_dt
        
        return trajectory


class AdjacencyNetwork(nn.Module):
    """
    Neural network that outputs an adjacency matrix from time series input.
    
    Architecture: Input (time series) -> Hidden layers -> Output (vectorized adjacency)
    """
    
    def __init__(
        self,
        n_nodes: int,
        input_length: int,
        hidden_layers: List[int] = [20, 20, 20, 20, 20],
        activation: str = "tanh",
        use_bias: bool = False,
        enforce_symmetry: bool = True,
        enforce_sparsity: bool = True,
    ):
        """
        Initialize adjacency network.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the network
        input_length : int
            Length of input time series (n_steps * n_nodes when flattened)
        hidden_layers : list of int, default [20, 20, 20, 20, 20]
            Number of units in each hidden layer
        activation : str, default "tanh"
            Activation function: "tanh", "relu", "sigmoid"
        use_bias : bool, default False
            Whether to use bias terms
        enforce_symmetry : bool, default True
            If True, enforce symmetric adjacency (undirected graph)
        enforce_sparsity : bool, default True
            If True, use hard sigmoid to allow exact zeros
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for neural network inference. "
                "Install with: pip install torch"
            )
        
        super().__init__()
        
        self.n_nodes = n_nodes
        self.input_length = input_length
        self.enforce_symmetry = enforce_symmetry
        self.enforce_sparsity = enforce_sparsity
        
        # Build network layers
        layers = []
        input_dim = input_length
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            input_dim = hidden_dim
        
        # Output layer: vectorized adjacency matrix (n_nodes * n_nodes)
        output_dim = n_nodes * n_nodes
        layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
        
        # Last activation: hard sigmoid for sparsity or standard sigmoid
        if enforce_sparsity:
            layers.append(nn.Hardsigmoid())
        else:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: time series -> adjacency matrix.
        
        Parameters
        ----------
        x : torch.Tensor (batch_size, input_length)
            Input time series
        
        Returns
        -------
        torch.Tensor (batch_size, n_nodes, n_nodes)
            Adjacency matrix
        """
        # Get vectorized adjacency
        A_vec = self.network(x)  # (batch_size, n_nodes * n_nodes)
        
        # Reshape to matrix
        batch_size = A_vec.shape[0]
        A = A_vec.view(batch_size, self.n_nodes, self.n_nodes)
        
        # Enforce symmetry if needed
        if self.enforce_symmetry:
            A = (A + A.transpose(-2, -1)) / 2
        
        # Zero diagonal (no self-loops)
        A = A.clone()
        for i in range(self.n_nodes):
            A[:, i, i] = 0
        
        return A


class NeuralNetworkInference:
    """
    Neural network-based inference of network adjacency matrices from time series.
    
    Trains a neural network to find an adjacency matrix that, when used in
    a dynamics model, reproduces the observed time series.
    """
    
    def __init__(
        self,
        n_nodes: int,
        dynamics_model: DynamicsModel,
        hidden_layers: List[int] = [20, 20, 20, 20, 20],
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        enforce_symmetry: bool = True,
        enforce_sparsity: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize neural network inference.
        
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the network
        dynamics_model : DynamicsModel
            Dynamics model to use for simulation
        hidden_layers : list of int, default [20, 20, 20, 20, 20]
            Hidden layer sizes
        learning_rate : float, default 1e-3
            Learning rate for optimizer
        optimizer : str, default "adam"
            Optimizer: "adam", "sgd", "rmsprop"
        enforce_symmetry : bool, default True
            Enforce symmetric adjacency (undirected)
        enforce_sparsity : bool, default True
            Enforce sparsity (allow exact zeros)
        device : str, optional
            Device to use ("cpu" or "cuda"). If None, auto-detects.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for neural network inference. "
                "Install with: pip install torch"
            )
        
        self.n_nodes = n_nodes
        self.dynamics_model = dynamics_model
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize network (will be created in fit)
        self.network = None
        self.optimizer = None
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.enforce_symmetry = enforce_symmetry
        self.enforce_sparsity = enforce_sparsity
        
        # Training history
        self.loss_history = []
        self.adjacency_history = []
    
    def fit(
        self,
        X: NDArray[np.float64],
        initial_conditions: Optional[NDArray[np.float64]] = None,
        n_epochs: int = 100,
        batch_size: int = 1,
        dt: float = 0.01,
        verbose: bool = True,
        **dynamics_params
    ) -> 'NeuralNetworkInference':
        """
        Fit the neural network to infer adjacency matrix.
        
        Parameters
        ----------
        X : array (n_steps, n_nodes)
            Observed time series
        initial_conditions : array (n_nodes,), optional
            Initial conditions. If None, uses first time step.
        n_epochs : int, default 100
            Number of training epochs
        batch_size : int, default 1
            Batch size for training
        dt : float, default 0.01
            Time step size for dynamics
        verbose : bool, default True
            Print training progress
        **dynamics_params
            Additional parameters for dynamics model
        
        Returns
        -------
        self : NeuralNetworkInference
        """
        X = np.asarray(X, dtype=np.float64)
        n_steps, n_nodes = X.shape
        
        if n_nodes != self.n_nodes:
            raise ValueError(f"Time series has {n_nodes} nodes, but model expects {self.n_nodes}")
        
        if initial_conditions is None:
            initial_conditions = X[0]
        
        # Initialize network if needed
        if self.network is None:
            # Input length is flattened time series: n_steps * n_nodes
            input_length = n_steps * n_nodes
            self.network = AdjacencyNetwork(
                n_nodes=self.n_nodes,
                input_length=input_length,
                hidden_layers=self.hidden_layers,
                enforce_symmetry=self.enforce_symmetry,
                enforce_sparsity=self.enforce_sparsity,
            ).to(self.device)
        
        # Initialize optimizer
        if self.optimizer is None:
            if self.optimizer_name.lower() == "adam":
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
            elif self.optimizer_name.lower() == "sgd":
                self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
            elif self.optimizer_name.lower() == "rmsprop":
                self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Convert to tensors
        # Input should be the full time series flattened: (n_steps * n_nodes,)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        X_input = X_tensor.flatten().unsqueeze(0)  # (1, n_steps * n_nodes)
        
        # Training loop
        self.loss_history = []
        self.adjacency_history = []
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # Get predicted adjacency
            A_pred = self.network(X_input)  # (1, n_nodes, n_nodes)
            A_pred_np = A_pred[0].detach().cpu().numpy()
            
            # Simulate dynamics
            X_sim = self.dynamics_model.forward(
                A=A_pred_np,
                initial_conditions=initial_conditions,
                n_steps=n_steps,
                dt=dt,
                **dynamics_params
            )
            X_sim_tensor = torch.from_numpy(X_sim).float().to(self.device)
            
            # Loss: MSE between observed and simulated
            loss = torch.mean((X_tensor - X_sim_tensor) ** 2)
            
            # Additional loss terms
            # Enforce symmetry (if not already enforced in network)
            if not self.enforce_symmetry:
                A_sym = (A_pred + A_pred.transpose(-2, -1)) / 2
                symmetry_loss = torch.mean((A_pred - A_sym) ** 2)
                loss += 0.1 * symmetry_loss
            
            # Zero diagonal
            diag_loss = torch.mean(A_pred[:, range(self.n_nodes), range(self.n_nodes)] ** 2)
            loss += 0.1 * diag_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Store history
            self.loss_history.append(loss.item())
            self.adjacency_history.append(A_pred_np.copy())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
        
        return self
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict adjacency matrix from time series.
        
        Parameters
        ----------
        X : array (n_steps, n_nodes)
            Time series
        
        Returns
        -------
        array (n_nodes, n_nodes)
            Predicted adjacency matrix
        """
        if self.network is None:
            raise ValueError("Model must be fitted first. Call fit() before predict().")
        
        X_tensor = torch.from_numpy(X).float().to(self.device)
        # Flatten time series for input
        X_input = X_tensor.flatten().unsqueeze(0)  # (1, n_steps * n_nodes)
        
        with torch.no_grad():
            A_pred = self.network(X_input)
        
        return A_pred[0].cpu().numpy()
    
    def get_uncertainty(self) -> Dict[str, NDArray[np.float64]]:
        """
        Get uncertainty quantification from training history.
        
        Returns
        -------
        dict
            Dictionary with:
            - 'mean': Mean adjacency matrix
            - 'std': Standard deviation of adjacency matrix
            - 'mode': Mode (adjacency at minimum loss)
        """
        if len(self.adjacency_history) == 0:
            raise ValueError("No training history available. Call fit() first.")
        
        # Convert to array
        A_history = np.array(self.adjacency_history)  # (n_epochs, n_nodes, n_nodes)
        
        # Find mode (minimum loss)
        min_loss_idx = np.argmin(self.loss_history)
        mode = A_history[min_loss_idx]
        
        # Compute statistics
        mean = np.mean(A_history, axis=0)
        std = np.std(A_history, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'mode': mode,
        }

