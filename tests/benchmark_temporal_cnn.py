"""
Benchmark for temporal CNN embeddings.

Compares speed and memory usage of CNN embeddings vs graph-based feature extraction.
"""

import time
import numpy as np
import pytest

try:
    from ts2net.temporal_cnn import temporal_cnn_embeddings
    from ts2net.api import HVG
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    pytestmark = pytest.mark.skip("PyTorch not installed")


@pytest.mark.benchmark
class TestTemporalCNNBenchmark:
    """Benchmark tests for temporal CNN."""
    
    def test_cnn_speed_small_series(self, benchmark):
        """Benchmark CNN on small series."""
        x = np.random.randn(1000)
        
        def run_cnn():
            return temporal_cnn_embeddings(x, window=50, stride=10, seed=42)
        
        result = benchmark(run_cnn)
        assert result.shape[0] > 0
        assert np.all(np.isfinite(result))
    
    def test_cnn_speed_large_series(self, benchmark):
        """Benchmark CNN on large series."""
        x = np.random.randn(10000)
        
        def run_cnn():
            return temporal_cnn_embeddings(x, window=100, stride=20, seed=42)
        
        result = benchmark(run_cnn)
        assert result.shape[0] > 0
    
    def test_cnn_vs_graph_features(self):
        """Compare CNN embeddings vs graph-based features (qualitative)."""
        x = np.random.randn(2000)
        window = 100
        stride = 20
        
        # CNN path
        start = time.time()
        cnn_embeddings = temporal_cnn_embeddings(x, window=window, stride=stride, seed=42)
        cnn_time = time.time() - start
        
        # Graph path (HVG stats per window)
        start = time.time()
        n_windows = (len(x) - window) // stride + 1
        graph_features = []
        for i in range(0, len(x) - window + 1, stride):
            window_data = x[i:i + window]
            hvg = HVG(output='stats')
            hvg.build(window_data)
            stats = hvg.stats()
            # Extract key features
            features = np.array([
                stats.get('avg_degree', 0),
                stats.get('std_degree', 0),
                stats.get('n_edges', 0) / stats.get('n_nodes', 1),
            ])
            graph_features.append(features)
        graph_features = np.array(graph_features)
        graph_time = time.time() - start
        
        # Both should produce features
        assert cnn_embeddings.shape[0] == graph_features.shape[0]
        assert cnn_embeddings.shape[0] == n_windows
        
        # Log comparison (not asserting, just informative)
        print(f"\nCNN: {cnn_time:.3f}s, shape={cnn_embeddings.shape}, "
              f"features per window={cnn_embeddings.shape[1]}")
        print(f"Graph: {graph_time:.3f}s, shape={graph_features.shape}, "
              f"features per window={graph_features.shape[1]}")
        print(f"Speedup: {graph_time/cnn_time:.2f}x")
    
    def test_cnn_memory_efficiency(self):
        """Test that CNN doesn't create excessive memory usage."""
        import tracemalloc
        
        x = np.random.randn(5000)
        
        tracemalloc.start()
        embeddings = temporal_cnn_embeddings(x, window=100, stride=20, seed=42)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should be reasonable (less than 100MB peak)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds 100MB"
        
        print(f"\nPeak memory: {peak_mb:.1f}MB")
        print(f"Output shape: {embeddings.shape}, size: {embeddings.nbytes / 1024 / 1024:.2f}MB")
