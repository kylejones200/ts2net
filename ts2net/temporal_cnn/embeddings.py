"""
Temporal CNN embeddings for time series windows.

Implements a 1D dilated CNN with causal padding and global average pooling.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def temporal_cnn_embeddings(
    x: NDArray[np.float64],
    window: int,
    stride: int,
    *,
    channels: Tuple[int, ...] = (32, 64, 64),
    kernel_size: int = 5,
    dilations: Tuple[int, ...] = (1, 2, 4),
    dropout: float = 0.1,
    device: str = "cpu",
    batch_size: int = 256,
    seed: int = 7,
) -> NDArray[np.float64]:
    """
    Compute per-window embeddings with a small dilated 1D CNN.

    Args:
        x: Array of shape (n, f) or (n,). For multivariate, f is number of features.
        window: Window length in time steps.
        stride: Step between windows.
        channels: Output channels per conv block. Length must match dilations.
        kernel_size: Kernel size for each conv.
        dilations: Dilation per block. Length must match channels length.
        dropout: Dropout rate.
        device: Torch device string ('cpu' or 'cuda').
        batch_size: Batch size for inference.
        seed: Random seed for determinism.

    Returns:
        Array of shape (n_windows, channels[-1]) with embeddings.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: If input shape is invalid or parameters don't match.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch required for temporal_cnn_embeddings. "
            "Install with: pip install torch"
        )
    
    # Convert to numpy array if needed
    x = np.asarray(x, dtype=np.float64)
    
    # Handle univariate vs multivariate
    if x.ndim == 1:
        x = x[:, np.newaxis]  # (n,) -> (n, 1)
    elif x.ndim != 2:
        raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")
    
    n, n_features = x.shape
    
    if n < window:
        raise ValueError(f"Series length {n} < window size {window}")
    
    if len(channels) != len(dilations):
        raise ValueError(
            f"channels length ({len(channels)}) must match dilations length ({len(dilations)})"
        )
    
    # Set random seed for determinism (before model creation)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Set CUDNN to deterministic mode if CUDA is available
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Extract windows
    windows = []
    for i in range(0, n - window + 1, stride):
        windows.append(x[i:i + window])
    
    if len(windows) == 0:
        return np.empty((0, channels[-1]), dtype=np.float64)
    
    # Convert to tensor: (n_windows, window, n_features)
    windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    
    # Transpose to (n_windows, n_features, window) for conv1d
    windows_tensor = windows_tensor.transpose(1, 2)
    
    # Build model
    model = _TemporalCNN(
        in_channels=n_features,
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
    ).to(device)
    
    model.eval()  # Set to evaluation mode
    
    # Process in batches
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(windows_tensor), batch_size):
            batch = windows_tensor[i:i + batch_size].to(device)
            batch_embeddings = model(batch)
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Ensure finite values
    if not np.all(np.isfinite(embeddings)):
        import warnings
        warnings.warn(
            "Non-finite values detected in embeddings. Replacing with zeros.",
            UserWarning
        )
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    
    return embeddings.astype(np.float64)


# Only define the class when PyTorch is available
if HAS_TORCH:
    class _TemporalCNN(nn.Module):
        """1D Temporal CNN with dilations and causal padding."""
        
        def __init__(
            self,
            in_channels: int,
            channels: Tuple[int, ...],
            kernel_size: int,
            dilations: Tuple[int, ...],
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.in_channels = in_channels
            self.channels = channels
            self.kernel_size = kernel_size
            self.dilations = dilations
            
            # Build conv blocks
            layers = []
            prev_channels = in_channels
            
            for out_channels, dilation in zip(channels, dilations):
                # Causal padding: (kernel_size - 1) * dilation
                padding = (kernel_size - 1) * dilation
                
                layers.append(
                    nn.Conv1d(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                        bias=True,
                    )
                )
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
                prev_channels = out_channels
            
            self.conv_blocks = nn.Sequential(*layers)
            
            # Global average pooling
            self.pool = nn.AdaptiveAvgPool1d(1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, in_channels, length)
                
            Returns:
                Embeddings of shape (batch, channels[-1])
            """
            # Apply conv blocks
            x = self.conv_blocks(x)
            
            # Global average pooling: (batch, channels[-1], length) -> (batch, channels[-1], 1)
            x = self.pool(x)
            
            # Squeeze: (batch, channels[-1], 1) -> (batch, channels[-1])
            x = x.squeeze(-1)
            
            return x
