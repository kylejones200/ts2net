"""
Benchmark script to demonstrate Numba acceleration in ts2net.

This script compares performance with and without Numba for various algorithms.
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Check if Numba is available
try:
    import numba
    HAS_NUMBA = True
    logger.info(f"Numba {numba.__version__} installed")
except ImportError:
    HAS_NUMBA = False
    logger.warning("Numba not installed. Install with: pip install numba")

# Import ts2net components
try:
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
    logger.info("ts2net imported")
except ImportError as e:
    logger.error(f"Failed to import ts2net: {e}")
    sys.exit(1)


def benchmark_hvg():
    """Benchmark HVG performance."""
    logger.info("HVG:")
    for n in [100, 500, 1000, 2000]:
        x = np.sin(np.linspace(0, 12 * np.pi, n)) + 0.1 * np.random.randn(n)
        if HAS_NUMBA:
            HVG().build(x)
        
        start = time.time()
        hvg = HVG().build(x)
        elapsed = time.time() - start
        
        logger.info("n=%d %.4fs nodes=%d edges=%d", n, elapsed, hvg.n_nodes, hvg.n_edges)


def benchmark_nvg():
    """Benchmark NVG performance."""
    logger.info("NVG:")
    for n in [50, 100, 200, 500]:
        x = np.sin(np.linspace(0, 12 * np.pi, n)) + 0.1 * np.random.randn(n)
        if HAS_NUMBA:
            NVG().build(x)
        
        start = time.time()
        nvg = NVG().build(x)
        elapsed = time.time() - start
        
        logger.info("n=%d %.4fs nodes=%d edges=%d", n, elapsed, nvg.n_nodes, nvg.n_edges)


def benchmark_recurrence():
    """Benchmark Recurrence Network performance."""
    logger.info("Recurrence:")
    for n, m, tau, metric in [(200, 2, 1, "euclidean"), (500, 3, 2, "euclidean"), 
                               (1000, 3, 2, "manhattan"), (2000, 2, 1, "chebyshev")]:
        x = np.sin(np.linspace(0, 12 * np.pi, n)) + 0.1 * np.random.randn(n)
        if HAS_NUMBA:
            RecurrenceNetwork(m=m, tau=tau, rule="knn", k=8, metric=metric).build(x)
        
        start = time.time()
        rn = RecurrenceNetwork(m=m, tau=tau, rule="knn", k=8, metric=metric).build(x)
        elapsed = time.time() - start
        
        logger.info("n=%d m=%d τ=%d %s %.4fs edges=%d", n, m, tau, metric, elapsed, rn.n_edges)


def benchmark_transition():
    """Benchmark Transition Network performance."""
    logger.info("Transition:")
    for n, order, delay in [(500, 3, 1), (1000, 3, 1), (2000, 4, 1), (5000, 3, 2)]:
        x = np.sin(np.linspace(0, 12 * np.pi, n)) + 0.1 * np.random.randn(n)
        if HAS_NUMBA:
            TransitionNetwork(symbolizer="ordinal", order=order).build(x)
        
        start = time.time()
        tn = TransitionNetwork(symbolizer="ordinal", order=order).build(x)
        elapsed = time.time() - start
        
        logger.info("n=%d order=%d %.4fs states=%d", n, order, elapsed, tn.n_nodes)


def main():
    """Run all benchmarks."""
    not HAS_NUMBA and logger.warning("Numba not installed - install for 10-200x speedups")
    
    for benchmark_fn in [benchmark_hvg, benchmark_nvg, benchmark_recurrence, benchmark_transition]:
        benchmark_fn()


if __name__ == "__main__":
    main()

