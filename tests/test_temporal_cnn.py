"""
Tests for temporal CNN embeddings.

Tests shape, determinism, finite values, and cross-run consistency.
"""

import numpy as np
import pytest

try:
    from ts2net.temporal_cnn import temporal_cnn_embeddings
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytestmark = pytest.mark.skip("PyTorch not installed")


class TestTemporalCNNShape:
    """Test output shapes."""
    
    def test_univariate_shape(self):
        """Test univariate input shape."""
        np.random.seed(42)
        x = np.random.randn(1000)
        embeddings = temporal_cnn_embeddings(x, window=50, stride=10)
        
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 64  # Default channels[-1]
        # n_windows = (1000 - 50) // 10 + 1 = 96
        assert embeddings.shape[0] == 96
    
    def test_multivariate_shape(self):
        """Test multivariate input shape."""
        np.random.seed(42)
        x = np.random.randn(1000, 3)  # 3 features
        embeddings = temporal_cnn_embeddings(x, window=50, stride=10)
        
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 64
        assert embeddings.shape[0] == 96
    
    def test_custom_channels(self):
        """Test custom channel configuration."""
        np.random.seed(42)
        x = np.random.randn(500)
        embeddings = temporal_cnn_embeddings(
            x, window=30, stride=5,
            channels=(16, 32, 48),
            dilations=(1, 2, 4)
        )
        
        assert embeddings.shape[1] == 48  # Last channel size
        assert embeddings.shape[0] == (500 - 30) // 5 + 1
    
    def test_stride_one(self):
        """Test stride=1 (overlapping windows)."""
        np.random.seed(42)
        x = np.random.randn(100)
        embeddings = temporal_cnn_embeddings(x, window=20, stride=1)
        
        assert embeddings.shape[0] == 100 - 20 + 1  # 81 windows
        assert embeddings.shape[1] == 64


class TestTemporalCNNDeterminism:
    """Test determinism and reproducibility."""
    
    def test_same_seed_same_output(self):
        """Test that same seed produces same output."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        embeddings1 = temporal_cnn_embeddings(x, window=30, stride=5, seed=42)
        embeddings2 = temporal_cnn_embeddings(x, window=30, stride=5, seed=42)
        
        np.testing.assert_array_equal(embeddings1, embeddings2)
    
    def test_different_seed_different_output(self):
        """Test that different seeds produce different output."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        embeddings1 = temporal_cnn_embeddings(x, window=30, stride=5, seed=42)
        embeddings2 = temporal_cnn_embeddings(x, window=30, stride=5, seed=43)
        
        # Should be different (very unlikely to be identical)
        assert not np.allclose(embeddings1, embeddings2, atol=1e-6)
    
    def test_deterministic_across_runs(self):
        """Test that output is deterministic across multiple runs."""
        np.random.seed(42)
        x = np.random.randn(150)
        
        # Run multiple times with same seed
        results = []
        for _ in range(3):
            emb = temporal_cnn_embeddings(x, window=25, stride=5, seed=7)
            results.append(emb)
        
        # All should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class TestTemporalCNNFiniteValues:
    """Test that outputs are finite."""
    
    def test_all_finite(self):
        """Test that all output values are finite."""
        np.random.seed(42)
        x = np.random.randn(300)
        embeddings = temporal_cnn_embeddings(x, window=40, stride=8)
        
        assert np.all(np.isfinite(embeddings))
        assert not np.any(np.isnan(embeddings))
        assert not np.any(np.isinf(embeddings))
    
    def test_finite_with_extreme_input(self):
        """Test finite output with extreme input values."""
        np.random.seed(42)
        x = np.random.randn(200) * 1000  # Large values
        embeddings = temporal_cnn_embeddings(x, window=30, stride=5)
        
        assert np.all(np.isfinite(embeddings))
    
    def test_finite_with_constant_input(self):
        """Test finite output with constant input."""
        x = np.ones(200) * 5.0
        embeddings = temporal_cnn_embeddings(x, window=30, stride=5)
        
        assert np.all(np.isfinite(embeddings))


class TestTemporalCNNParameters:
    """Test parameter validation and edge cases."""
    
    def test_short_series_error(self):
        """Test error for series shorter than window."""
        np.random.seed(42)
        x = np.random.randn(10)
        
        with pytest.raises(ValueError, match="Series length"):
            temporal_cnn_embeddings(x, window=20, stride=5)
    
    def test_channel_dilation_mismatch_error(self):
        """Test error when channels and dilations don't match."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        with pytest.raises(ValueError, match="channels length"):
            temporal_cnn_embeddings(
                x, window=30, stride=5,
                channels=(32, 64),
                dilations=(1, 2, 4)  # Mismatch
            )
    
    def test_multivariate_different_features(self):
        """Test with different numbers of features."""
        np.random.seed(42)
        for n_features in [1, 2, 5, 10]:
            x = np.random.randn(200, n_features)
            embeddings = temporal_cnn_embeddings(x, window=30, stride=5)
            
            assert embeddings.shape[0] > 0
            assert embeddings.shape[1] == 64
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        for kernel_size in [3, 5, 7]:
            embeddings = temporal_cnn_embeddings(
                x, window=30, stride=5,
                kernel_size=kernel_size
            )
            assert embeddings.shape[0] > 0
            assert embeddings.shape[1] == 64
    
    def test_different_dilations(self):
        """Test with different dilation schedules."""
        np.random.seed(42)
        x = np.random.randn(200)
        
        dilations_configs = [
            (1, 1, 1),  # No dilation
            (1, 2, 4),  # Default
            (1, 3, 9),  # Larger dilations
        ]
        
        for dilations in dilations_configs:
            channels = (32, 64, 64)
            embeddings = temporal_cnn_embeddings(
                x, window=30, stride=5,
                channels=channels,
                dilations=dilations
            )
            assert embeddings.shape[0] > 0
            assert embeddings.shape[1] == channels[-1]


class TestTemporalCNNBatchProcessing:
    """Test batch processing."""
    
    def test_small_batch_size(self):
        """Test with small batch size."""
        np.random.seed(42)
        x = np.random.randn(1000)
        embeddings = temporal_cnn_embeddings(
            x, window=50, stride=10,
            batch_size=10
        )
        
        assert embeddings.shape[0] == (1000 - 50) // 10 + 1
    
    def test_large_batch_size(self):
        """Test with large batch size."""
        np.random.seed(42)
        x = np.random.randn(500)
        embeddings = temporal_cnn_embeddings(
            x, window=30, stride=5,
            batch_size=1000  # Larger than number of windows
        )
        
        assert embeddings.shape[0] == (500 - 30) // 5 + 1


class TestTemporalCNNDevice:
    """Test device handling."""
    
    def test_cpu_device(self):
        """Test explicit CPU device."""
        np.random.seed(42)
        x = np.random.randn(200)
        embeddings = temporal_cnn_embeddings(
            x, window=30, stride=5,
            device="cpu"
        )
        
        assert embeddings.shape[0] > 0
        assert np.all(np.isfinite(embeddings))


class TestTemporalCNNIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_sine_wave(self):
        """Test with sine wave (common time series pattern)."""
        np.random.seed(42)
        t = np.linspace(0, 4*np.pi, 500)
        x = np.sin(t) + 0.1 * np.random.randn(500)
        
        embeddings = temporal_cnn_embeddings(x, window=50, stride=10)
        
        assert embeddings.shape[0] > 0
        assert np.all(np.isfinite(embeddings))
        # Embeddings should capture some structure
        assert np.std(embeddings) > 0
    
    def test_multivariate_sine(self):
        """Test multivariate sine waves."""
        t = np.linspace(0, 4*np.pi, 400)
        x = np.column_stack([
            np.sin(t),
            np.cos(t),
            np.sin(2*t),
        ])
        
        embeddings = temporal_cnn_embeddings(x, window=40, stride=8)
        
        assert embeddings.shape[0] > 0
        assert embeddings.shape[1] == 64
        assert np.all(np.isfinite(embeddings))
    
    def test_windowed_pipeline_compatible(self):
        """Test that output can be used in windowed pipeline."""
        np.random.seed(42)
        x = np.random.randn(1000)
        
        # Extract embeddings
        embeddings = temporal_cnn_embeddings(x, window=50, stride=25)
        
        # Should be compatible with downstream processing
        assert embeddings.ndim == 2
        assert embeddings.shape[0] > 0
        assert embeddings.dtype == np.float64
