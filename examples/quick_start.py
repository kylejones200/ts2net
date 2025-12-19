"""Quick Start Example

This example demonstrates basic usage with synthetic data.
For real-world examples, see example_fred_data.py
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from ts2net import HVG, build_network

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Generate synthetic time series (random walk with trend)
np.random.seed(42)
n = 1000
x = np.cumsum(np.random.randn(n)) + 0.01 * np.arange(n)  # Random walk with drift

logger.info("ts2net Quick Start\n")

logger.info("Basic HVG")
hvg = HVG()
hvg.build(x)
logger.info("Nodes: %d, Edges: %d", hvg.n_nodes, hvg.n_edges)
logger.info("Avg degree: %.2f\n", np.mean(hvg.degree_sequence()))

logger.info("Adjacency Matrix")
A = hvg.adjacency_matrix()
density = hvg.n_edges / (hvg.n_nodes * (hvg.n_nodes-1) / 2)
logger.info("Shape: %s, Density: %.4f\n", A.shape, density)

logger.info("NetworkX Conversion")
G_nx = hvg.as_networkx()
logger.info("Type: %s\n", type(G_nx).__name__)

logger.info("Performance Mode (only_degrees=True)")
hvg_fast = HVG(only_degrees=True)
hvg_fast.build(x)
degrees = hvg_fast.degree_sequence()
logger.info("Degrees: %d, Edges stored: %s\n", len(degrees), hvg_fast.edges)

logger.info("Compare Methods")
for method in ['hvg', 'nvg', 'recurrence', 'transition']:
    if method == 'recurrence':
        g = build_network(x, method, m=3, rule='knn', k=5)
    elif method == 'transition':
        g = build_network(x, method, symbolizer='ordinal', order=3)
    else:
        g = build_network(x, method)
    logger.info("%s: %d edges", method, g.n_edges)
