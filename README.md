# ts2net

[![PyPI version](https://badge.fury.io/py/ts2net.svg)](https://badge.fury.io/py/ts2net)
[![Documentation Status](https://readthedocs.org/projects/ts2net/badge/?version=latest)](https://ts2net.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/kylejones200/ts2net/workflows/Tests/badge.svg)](https://github.com/kylejones200/ts2net/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Time series to networks. Clean API for visibility graphs, recurrence networks, and transition networks.

## Install

### Basic Installation

```bash
pip install ts2net
```

### Optional Dependencies

Install with optional features:

```bash
# Performance acceleration (Numba)
pip install ts2net[speed]

# BSTS decomposition (statsmodels)
pip install ts2net[bsts]

# Temporal CNN embeddings (PyTorch)
pip install ts2net[cnn]

# All optional features
pip install ts2net[all]

# Development dependencies
pip install ts2net[dev]
```

### Verify Installation

```python
import ts2net
print(ts2net.__version__)

from ts2net import HVG
import numpy as np
x = np.random.randn(100)
hvg = HVG()
hvg.build(x)
print(f"✓ Installation successful: {hvg.n_nodes} nodes, {hvg.n_edges} edges")
```

## Quick Start

```python
import numpy as np
from ts2net import HVG

x = np.random.randn(1000)

hvg = HVG()
hvg.build(x)

print(hvg.n_nodes, hvg.n_edges)
print(hvg.degree_sequence())
```

## Adjacency Matrix

```python
A = hvg.adjacency_matrix()
print(A.shape)  # (1000, 1000)
```

## NetworkX (Optional)

NetworkX is optional. Convert only if needed:

```python
G = hvg.as_networkx()
import networkx as nx
print(nx.average_clustering(G))
```

## Structural Decomposition and Residual Topology

For time series with predictable structure (seasonality, trends), decompose first, then analyze the residual:

```python
from ts2net.bsts import features, BSTSSpec

# Decompose and analyze residual in one pass
spec = BSTSSpec(
    level=True,
    trend=False,
    seasonal_periods=[24, 168]  # Daily and weekly for hourly data
)

result = features(x, methods=['hvg', 'transition'], bsts=spec)

# Access three feature blocks
raw_stats = result.raw_stats              # Basic series statistics
structural_stats = result.structural_stats # Component variances, seasonal strength
residual_network_stats = result.residual_network_stats  # Network features from residual
```

**Use cases:**
- Compare meters/wells without seasonal confounds
- Flag series where structural model fails (high residual complexity)
- Separate predictable structure from irregular dynamics

**Installation:**
```bash
pip install ts2net[bsts]  # Installs statsmodels
```

See `examples/bsts_features.py` for complete examples.

## Large Series

For large series, use output modes to control memory usage:

```python
# Degrees only (most memory efficient)
hvg = HVG(output="degrees")
hvg.build(x)
degrees = hvg.degree_sequence()  # Fast, no edge storage

# Stats only (summary statistics without edges)
hvg = HVG(output="stats")
hvg.build(x)
stats = hvg.stats()  # n_nodes, n_edges, avg_degree, etc.

# Full edges (default, use for small-medium series)
hvg = HVG(output="edges")
hvg.build(x)
edges = hvg.edges  # Full edge list
```

### Scale Guidelines

| Series Length | Method | Recommended Settings | Memory Risk |
|--------------|--------|---------------------|-------------|
| n < 10k | All methods | `output="edges"` | Safe |
| 10k < n < 100k | HVG | `output="edges"` or `output="degrees"` | Safe with sparse |
| 10k < n < 100k | NVG | `limit=2000-5000`, `output="degrees"` | **Use horizon limit** |
| 10k < n < 100k | Recurrence | `rule='knn'`, `k=10-30` | **Avoid exact all-pairs** |
| n > 100k | HVG | `output="degrees"` or `output="stats"` | Safe |
| n > 100k | NVG | `limit=2000-5000`, `max_edges=1e6`, `output="degrees"` | **Required limits** |
| n > 100k | Recurrence | `rule='knn'`, `k=10-30`, `output="degrees"` | **kNN only** |

**Critical Warnings:**
- **Dense adjacency matrices are disabled by default** for n > 50k (prevents 63GB+ memory blowup)
- **NVG without `limit` can create millions of edges** for smooth series
- **Recurrence exact all-pairs is O(n²) memory** - use kNN for large n
- **NetworkX conversion refused** for n > 200k (use `force=True` to override)

**Memory Estimates:**
- Dense adjacency: ~8 * n² bytes (e.g., 90k nodes = 63 GB)
- Sparse adjacency: ~16 * m bytes where m = edges (e.g., 100k edges = 1.6 MB)
- Edge list: ~16 * m bytes (similar to sparse)
- Degrees only: ~8 * n bytes (e.g., 90k nodes = 720 KB)

## Methods

### Visibility Graphs

**HVG** - Horizontal Visibility Graph
```python
from ts2net import HVG

hvg = HVG(weighted=False, limit=None)
hvg.build(x)
```

**NVG** - Natural Visibility Graph
```python
from ts2net import NVG

# For large series, use horizon limit and bounded work
nvg = NVG(weighted=False, limit=5000, max_edges=1_000_000, output="degrees")
nvg.build(x)

# Or with memory limit
nvg = NVG(limit=2000, max_memory_mb=100)  # Caps at ~100MB
nvg.build(x)
```

### Recurrence Networks

Phase space recurrence:

```python
from ts2net import RecurrenceNetwork

rn = RecurrenceNetwork(m=3, tau=1, rule='knn', k=5)
rn.build(x)
```

Parameters:
- `m`: embedding dimension (None = auto via FNN)
- `tau`: time delay
- `rule`: 'knn', 'epsilon', 'radius'
- `k`: neighbors for k-NN
- `epsilon`: threshold for epsilon-recurrence

### Transition Networks

Symbolic dynamics:

```python
from ts2net import TransitionNetwork

tn = TransitionNetwork(symbolizer='ordinal', order=3)
tn.build(x)
```

Symbolizers:
- `'ordinal'`: ordinal patterns
- `'equal_width'`: equal-width bins
- `'equal_freq'`: equal-frequency bins (quantiles)
- `'kmeans'`: k-means clustering

## Compare Methods

```python
from ts2net import build_network

x = np.random.randn(1000)

for method in ['hvg', 'nvg', 'recurrence', 'transition']:
    if method == 'recurrence':
        g = build_network(x, method, m=3, rule='knn', k=5)
    elif method == 'transition':
        g = build_network(x, method, symbolizer='ordinal', order=3)
    else:
        g = build_network(x, method)
    
    print(f"{method}: {g.n_edges} edges")
```

Output:
```
hvg: 1979 edges
nvg: 2931 edges
recurrence: 3159 edges
transition: 18 edges
```

## Multivariate

Multiple time series → network where nodes = time series:

```python
from ts2net.multivariate import ts_dist, net_knn

X = np.random.randn(30, 1000)  # 30 series, 1000 points each

D = ts_dist(X, method='dtw', n_jobs=-1)
G = net_knn(D, k=5)

print(G.n_nodes, G.n_edges)
```

Distance methods: `'correlation'`, `'dtw'`, `'nmi'`, `'voi'`, `'es'`, `'vr'`

Network builders: `net_knn`, `net_enn`, `net_weighted`

## Performance

With Numba (recommended):

```bash
pip install numba
```

Speedups:
- HVG: 100x faster
- NVG: 180x faster
- Recurrence: 10x faster

## API

All methods follow the same pattern:

```python
builder = Method(**params)
builder.build(x)

# Access results
builder.n_nodes
builder.n_edges
builder.edges                # list of tuples
builder.degree_sequence()    # numpy array
builder.adjacency_matrix()   # numpy array
builder.as_networkx()        # optional conversion
```

## Troubleshooting

### Common Issues

**Memory errors with large time series:**
- Use `output="degrees"` or `output="stats"` instead of `output="edges"`
- For NVG, always set `limit` parameter (e.g., `limit=5000`)
- For recurrence networks, use `rule='knn'` with small `k` (10-30) instead of exact all-pairs

**Slow performance:**
- Install Numba: `pip install numba` (100-180x speedup for visibility graphs)
- Use `output="degrees"` if you don't need full edge lists
- For multivariate, use `n_jobs=-1` for parallel distance computation

**Import errors:**
- Ensure you're using Python 3.12+
- If Rust extension fails to build, install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- For development, run `maturin develop --release` after installing Rust

**NetworkX conversion refused:**
- For very large graphs (n > 200k), NetworkX conversion is disabled by default
- Use `force=True` to override: `hvg.as_networkx(force=True)`
- Consider using `output="degrees"` or `output="stats"` instead

### Getting Help

- 📖 [Full Documentation](https://ts2net.readthedocs.io)
- 💬 [Open an Issue](https://github.com/kylejones200/ts2net/issues)
- 📝 [Examples](https://github.com/kylejones200/ts2net/tree/main/examples)

## Citation

Multivariate methods based on:

Ferreira, L.N. (2024). From time series to networks in R with the ts2net package. *Applied Network Science*, 9(1), 32. https://doi.org/10.1007/s41109-024-00642-2

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

