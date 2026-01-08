# ts2net Quick Reference

## Installation

```bash
pip install ts2net                    # Basic
pip install ts2net[speed]             # With Numba (100-180x faster)
pip install ts2net[bsts]              # With BSTS decomposition
pip install ts2net[cnn]               # With Temporal CNN
pip install ts2net[all]               # All optional features
```

## Basic Usage

```python
from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
import numpy as np

x = np.random.randn(1000)

# Build graph
hvg = HVG()
hvg.build(x)

# Access results
hvg.n_nodes              # Number of nodes
hvg.n_edges              # Number of edges
hvg.edges                # List of (i, j) tuples
hvg.degree_sequence()    # NumPy array of degrees
hvg.adjacency_matrix()   # Sparse or dense adjacency matrix
hvg.as_networkx()        # NetworkX graph (optional)
```

## Graph Builders

### HVG (Horizontal Visibility Graph)
```python
hvg = HVG(weighted=False, output="edges")
hvg.build(x)
```

### NVG (Natural Visibility Graph)
```python
nvg = NVG(weighted=False, limit=5000, output="degrees")
nvg.build(x)
```

### Recurrence Network
```python
rn = RecurrenceNetwork(m=3, tau=1, rule='knn', k=5)
rn.build(x)
```

### Transition Network
```python
tn = TransitionNetwork(symbolizer='ordinal', order=3)
tn.build(x)
```

## Output Modes

| Mode | Memory | Speed | Use Case |
|------|--------|-------|----------|
| `"edges"` | High | Medium | Small-medium series, need full graph |
| `"degrees"` | Low | Fast | Large series, only need degree stats |
| `"stats"` | Lowest | Fastest | Very large series, summary only |

## Scale Guidelines

| Series Length | Recommended Settings |
|--------------|---------------------|
| n < 10k | `output="edges"`, all methods |
| 10k < n < 100k | `output="degrees"`, NVG needs `limit=2000-5000` |
| n > 100k | `output="degrees"` or `"stats"`, NVG requires `limit` |

## Multivariate

```python
from ts2net.multivariate import ts_dist, net_knn

X = np.random.randn(30, 1000)  # 30 series, 1000 points
D = ts_dist(X, method='correlation', n_jobs=-1)
G = net_knn(D, k=5)
```

## Visualization

```python
from ts2net.viz import (
    plot_series_with_events,
    plot_degree_profile,
    plot_degree_ccdf,
    plot_method_comparison,
    plot_window_feature_map,
)
```

## Unified Graph API

```python
from ts2net.viz import (
    build_recurrence_graph,
    build_ordinal_partition_graph,
    build_visibility_graph,
    optimal_lag,
    optimal_dim,
)
```

## Performance Tips

1. **Install Numba**: `pip install numba` (100-180x speedup)
2. **Use appropriate output mode**: `"degrees"` for large series
3. **Set limits for NVG**: Always use `limit` parameter
4. **Use kNN for recurrence**: `rule='knn'` instead of exact all-pairs
5. **Parallel distance computation**: `n_jobs=-1` for multivariate

## Common Patterns

```python
# Compare methods
from ts2net import build_network
for method in ['hvg', 'nvg', 'recurrence', 'transition']:
    g = build_network(x, method, **kwargs)
    print(f"{method}: {g.n_edges} edges")

# Windowed analysis
from ts2net.api_windows import build_windows
windows = build_windows(x, window_size=100, step=50)
graphs = [HVG().build(w) for w in windows]

# BSTS decomposition
from ts2net.bsts import features, BSTSSpec
spec = BSTSSpec(level=True, seasonal_periods=[24, 168])
result = features(x, methods=['hvg'], bsts=spec)
```



