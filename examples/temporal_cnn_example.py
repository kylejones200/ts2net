"""
Example: Temporal CNN Embeddings

Demonstrates the 1D dilated CNN for fast feature extraction from time series.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from ts2net.temporal_cnn import temporal_cnn_embeddings
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def example_basic_embeddings():
    """Basic CNN embeddings example."""
    if not HAS_TORCH:
        logger.warning("Skipping example: PyTorch not installed")
        return
    
    logger.info("Example 1: Basic CNN Embeddings")
    
    # Create a simple time series
    t = np.linspace(0, 4*np.pi, 500)
    x = np.sin(t) + 0.1 * np.random.randn(500)
    
    # Extract embeddings
    embeddings = temporal_cnn_embeddings(
        x,
        window=50,
        stride=10,
        seed=42
    )
    
    logger.info(f"Input: {len(x)} points")
    logger.info(f"Output: {embeddings.shape[0]} windows, {embeddings.shape[1]} features per window")
    logger.info(f"All finite: {np.all(np.isfinite(embeddings))}")


def example_multivariate():
    """Multivariate time series embeddings."""
    if not HAS_TORCH:
        logger.warning("Skipping example: PyTorch not installed")
        return
    
    logger.info("Example 2: Multivariate Time Series")
    
    # Create multivariate series (e.g., 3 sensors)
    t = np.linspace(0, 4*np.pi, 400)
    x = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t) + 0.1 * np.random.randn(400),
    ])
    
    embeddings = temporal_cnn_embeddings(
        x,
        window=40,
        stride=8,
        seed=42
    )
    
    logger.info(f"Input: {x.shape[0]} points, {x.shape[1]} features")
    logger.info(f"Output: {embeddings.shape[0]} windows, {embeddings.shape[1]} features per window")


def example_custom_architecture():
    """Custom CNN architecture."""
    if not HAS_TORCH:
        logger.warning("Skipping example: PyTorch not installed")
        return
    
    logger.info("Example 3: Custom Architecture")
    
    x = np.random.randn(300)
    
    # Custom: smaller channels, different dilations
    embeddings = temporal_cnn_embeddings(
        x,
        window=30,
        stride=5,
        channels=(16, 32, 48),
        kernel_size=3,
        dilations=(1, 3, 9),
        dropout=0.0,  # No dropout for inference
        seed=42
    )
    
    logger.info(f"Custom architecture: {embeddings.shape[1]} features per window")
    logger.info(f"  Channels: (16, 32, 48)")
    logger.info(f"  Dilations: (1, 3, 9)")


def example_determinism():
    """Demonstrate determinism."""
    if not HAS_TORCH:
        logger.warning("Skipping example: PyTorch not installed")
        return
    
    logger.info("Example 4: Determinism Check")
    
    x = np.random.randn(200)
    
    # Run twice with same seed
    emb1 = temporal_cnn_embeddings(x, window=30, stride=5, seed=7)
    emb2 = temporal_cnn_embeddings(x, window=30, stride=5, seed=7)
    
    # Should be identical
    is_identical = np.allclose(emb1, emb2, atol=1e-6)
    logger.info(f"Same seed produces identical output: {is_identical}")
    
    # Different seed produces different output
    emb3 = temporal_cnn_embeddings(x, window=30, stride=5, seed=8)
    is_different = not np.allclose(emb1, emb3, atol=1e-6)
    logger.info(f"Different seed produces different output: {is_different}")


if __name__ == "__main__":
    if not HAS_TORCH:
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    logger.info("Temporal CNN Embeddings Examples")
    
    example_basic_embeddings()
    example_multivariate()
    example_custom_architecture()
    example_determinism()
    
    logger.info("All examples completed!")
