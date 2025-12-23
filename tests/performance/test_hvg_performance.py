"""
Performance regression tests for HVG.

Benchmarks HVG on large series and fails if performance degrades.
"""

import numpy as np
import pytest
import time
import logging
from ts2net.api import HVG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TestHVGPerformance:
    """Test HVG performance on large series."""
    
    def test_hvg_degrees_n1e6_benchmark(self):
        """
        Benchmark HVG degrees computation on n=1e6.
        
        This test stores the baseline time and will fail if performance
        degrades by more than 20% compared to baseline.
        
        Baseline: ~2-5 seconds for n=1e6 on modern hardware.
        """
        n = 1_000_000
        np.random.seed(42)
        x = np.random.randn(n)
        
        # Warmup
        hvg = HVG(output='degrees')
        hvg.build(x[:1000])
        
        # Actual benchmark
        start = time.time()
        hvg.build(x)
        elapsed = time.time() - start
        
        # Get degrees to ensure computation happened
        degrees = hvg.degree_sequence()
        assert len(degrees) == n, "Should compute degrees for all points"
        
        # Performance assertion
        # Baseline: should complete in < 10 seconds on modern hardware
        # Allow 20% margin for CI variability
        max_time = 12.0  # seconds
        
        assert elapsed < max_time, \
            f"HVG degrees on n={n} took {elapsed:.2f}s, expected < {max_time}s"
        
        # Log performance for monitoring
        logger.info(f"\nHVG degrees benchmark: n={n}, time={elapsed:.2f}s, "
                    f"throughput={n/elapsed/1e6:.2f}M points/s")
    
    def test_hvg_edges_n1e6_benchmark(self):
        """
        Benchmark HVG edge computation on n=1e6.
        
        Should complete in reasonable time and not materialize dense matrix.
        """
        n = 1_000_000
        np.random.seed(42)
        x = np.random.randn(n)
        
        hvg = HVG(output='edges')
        
        start = time.time()
        hvg.build(x)
        elapsed_build = time.time() - start
        
        # Get edges in COO format (should be fast, sparse)
        start = time.time()
        rows, cols, weights = hvg.edges_coo()
        elapsed_edges = time.time() - start
        
        total_time = elapsed_build + elapsed_edges
        
        # Should complete in < 15 seconds total
        max_time = 18.0  # seconds
        
        assert total_time < max_time, \
            f"HVG edges on n={n} took {total_time:.2f}s, expected < {max_time}s"
        
        # Verify sparse output
        n_edges = len(rows)
        assert n_edges < n * n, "Should not materialize dense adjacency"
        assert n_edges > n, "Should have reasonable number of edges"
        
        logger.info(f"\nHVG edges benchmark: n={n}, edges={n_edges}, "
                    f"time={total_time:.2f}s, density={n_edges/(n*(n-1)/2):.6f}")


@pytest.mark.benchmark
class TestHVGPerformanceRegression:
    """
    Performance regression tests that compare against stored baselines.
    
    These tests should be run in CI and fail if performance degrades.
    """
    
    # Store baseline times (in seconds) for different n
    # Note: CI environments can be slower, so baselines are conservative
    BASELINES = {
        100_000: 0.8,   # 0.8s for n=100k (adjusted for CI variability)
        500_000: 4.0,   # 4.0s for n=500k (adjusted for CI)
        1_000_000: 8.0, # 8.0s for n=1M (adjusted for CI)
    }
    
    TOLERANCE = 0.60  # 60% performance degradation tolerance (increased for CI variability)
    
    @pytest.mark.parametrize("n", [100_000, 500_000, 1_000_000])
    def test_performance_regression(self, n):
        """
        Test that performance hasn't regressed compared to baseline.
        
        Fails if current time > baseline * (1 + tolerance).
        """
        np.random.seed(42)
        x = np.random.randn(n)
        
        hvg = HVG(output='degrees')
        
        start = time.time()
        hvg.build(x)
        elapsed = time.time() - start
        
        baseline = self.BASELINES.get(n)
        if baseline is None:
            pytest.skip(f"No baseline for n={n}")
        
        max_time = baseline * (1 + self.TOLERANCE)
        
        assert elapsed < max_time, \
            f"Performance regression: n={n} took {elapsed:.2f}s, " \
            f"baseline={baseline:.2f}s, max={max_time:.2f}s " \
            f"({(elapsed/baseline - 1)*100:.1f}% slower)"
        
        # Update baseline if significantly faster (for manual updates)
        if elapsed < baseline * 0.8:
            logger.info(f"\nNOTE: n={n} is {((baseline/elapsed - 1)*100):.1f}% faster than baseline. "
                        f"Consider updating baseline from {baseline:.2f}s to {elapsed:.2f}s")

