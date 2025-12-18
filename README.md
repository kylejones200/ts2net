# ts2net

[![PyPI version](https://badge.fury.io/py/ts2net.svg)](https://badge.fury.io/py/ts2net)
[![Documentation Status](https://readthedocs.org/projects/ts2net/badge/?version=latest)](https://ts2net.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/kylejones200/ts2net/workflows/Tests/badge.svg)](https://github.com/kylejones200/ts2net/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Time series to networks. Clean API for visibility graphs, recurrence networks, and transition networks.

## Install

```bash
pip install ts2net
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

## Large Series

For large series (n > 100k), use `only_degrees=True` to skip edge storage:

```python
hvg = HVG(only_degrees=True)
hvg.build(x)

degrees = hvg.degree_sequence()  # Fast
# hvg.edges is None (not stored)
```

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

nvg = NVG(weighted=False, limit=None)
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

## Citation

Multivariate methods based on:

Ferreira, L.N. (2024). From time series to networks in R with the ts2net package. *Applied Network Science*, 9(1), 32. https://doi.org/10.1007/s41109-024-00642-2

## License

MIT

