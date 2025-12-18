# R ts2net API Mapping & Parity Specification

## R Function Signatures (from RDocumentation)

### Visibility Graphs

```r
tsnet_vg(x, method = c("horizontal", "natural"), limit = NULL, weighted = FALSE)
```

**Parameters:**
- `x`: numeric vector (time series)
- `method`: "horizontal" or "natural" (default: "horizontal")
- `limit`: integer, temporal visibility limit (default: NULL = no limit)
- `weighted`: logical, edge weights by value difference (default: FALSE)

**Returns:** igraph object (undirected)

### Recurrence Networks

```r
tsnet_rn(x, embedding.dim = NULL, time.lag = 1, radius = NULL, 
         k = NULL, metric = c("euclidean", "maximum", "manhattan"), 
         theiler.window = 0)
```

**Parameters:**
- `x`: numeric vector
- `embedding.dim`: integer, defaults to FNN selection if NULL
- `time.lag`: integer (default: 1)
- `radius`: numeric, ε-neighborhood threshold (for ε-RN)
- `k`: integer, k-nearest neighbors (for k-NN RN)
- `metric`: "euclidean", "maximum" (Chebyshev), "manhattan" (default: "euclidean")
- `theiler.window`: integer, temporal exclusion window (default: 0)

**Returns:** igraph object (undirected)

### Transition Networks

```r
tsnet_tn(x, order = 3, delay = 1, method = c("ordinal", "quantiles"))
```

**Parameters:**
- `x`: numeric vector
- `order`: integer, pattern length (default: 3)
- `delay`: integer, time delay (default: 1)
- `method`: "ordinal" for permutation patterns, "quantiles" for binning

**Returns:** igraph object (directed)

## Python API Alignment

### Current Python APIs

```python
# core/visibility/
HVG(weighted=False, sparse=False)
NVG(weighted=False, sparse=False)

# Missing: limit parameter
```

```python
# core/recurrence.py
RecurrenceNetwork(
    m=2,              # embedding.dim
    tau=1,            # time.lag
    rule="epsilon",   # determines radius vs k
    epsilon=None,     # radius
    k=10,             # k
    metric="euclidean",  # euclidean|manhattan|chebyshev
    theiler=0,        # theiler.window
    ...
)

# Issue: No auto embedding.dim selection
# Issue: metric "chebyshev" vs R "maximum"
```

```python
# core/transition.py
TransitionNetwork(
    symbolizer="ordinal",  # method
    order=3,              # order
    delay=1,              # delay
    ...
)

# Issue: "quantiles" not implemented
```

## Required Changes

### 1. Visibility Graphs - Add `limit` parameter

```python
class HVG:
    def __init__(self, weighted=False, sparse=False, limit=None):
        self.weighted = weighted
        self.sparse = sparse
        self.limit = limit  # NEW: temporal visibility limit
```

**Semantics:** Only connect points within temporal distance ≤ limit.
For HVG: only check visibility between i and j if |i-j| ≤ limit.

### 2. Recurrence Networks - Auto embedding dimension

```python
class RecurrenceNetwork:
    def __init__(
        self, 
        m=None,  # Changed: None triggers FNN-based selection
        tau=1,
        ...
    ):
        if m is None:
            # Implement FNN (False Nearest Neighbors) selection
            self.m = _auto_select_embedding_dim(x, tau)
```

**Action:** Implement FNN or disable auto-selection, document explicitly.

### 3. Metric name alignment

```python
# R: "maximum" → Python: "chebyshev"
_METRIC_ALIASES = {
    "maximum": "chebyshev",
    "chebyshev": "chebyshev",
}
```

### 4. Transition Networks - Add quantiles method

```python
TransitionNetwork(
    symbolizer="ordinal"|"quantiles"|"equal_width"|"equal_freq"|"kmeans",
    ...
)
```

**R "quantiles" = Python "equal_freq"** (need to verify exact behavior)

## Preprocessing Rules

### Input Validation

```python
def _validate_timeseries(x: np.ndarray) -> np.ndarray:
    """Apply consistent preprocessing rules."""
    x = np.asarray(x, dtype=np.float64).squeeze()
    
    # Rule 1: Must be 1D
    if x.ndim != 1:
        raise ValueError("Input must be 1D array")
    
    # Rule 2: Handle NA/NaN
    if np.any(np.isnan(x)):
        # Option A: Raise error (strict parity)
        raise ValueError("NaN values not allowed")
        # Option B: Drop NaN (R na.omit behavior)
        # x = x[~np.isnan(x)]
    
    # Rule 3: Handle Inf
    if np.any(np.isinf(x)):
        raise ValueError("Inf values not allowed")
    
    # Rule 4: Minimum length
    if len(x) < 3:
        raise ValueError("Series too short (min length: 3)")
    
    # Rule 5: Constant series
    if np.allclose(x, x[0]):
        raise ValueError("Constant series not allowed")
    
    return x
```

### Tie Handling

**HVG/NVG:**
- R uses `<` comparisons (strict inequality)
- Python must match: use `x[k] >= threshold` not `x[k] > threshold`

**Ordinal patterns:**
- R: stable sort preserves order
- Python: use `kind='stable'` in argsort or explicit tie-breaking

### Integer vs Float

Always cast to float64:
```python
x = np.asarray(x, dtype=np.float64)
```

## Parity Test Corpus

### Test Series Generators

```python
def generate_test_corpus(seed=42):
    """Generate deterministic test series."""
    rng = np.random.RandomState(seed)
    
    corpus = {
        # Basic patterns
        "sine_clean": lambda n: np.sin(np.linspace(0, 4*np.pi, n)),
        "sine_noise": lambda n: np.sin(np.linspace(0, 4*np.pi, n)) + 0.1*rng.randn(n),
        
        # Random processes
        "random_walk": lambda n: np.cumsum(rng.randn(n)),
        "ar1": lambda n: _generate_ar1(n, 0.7, rng),
        
        # Edge cases
        "constant_segments": lambda n: np.concatenate([
            np.ones(n//3), 
            2*np.ones(n//3), 
            3*np.ones(n//3)
        ]),
        "repeated_values": lambda n: rng.choice([1, 2, 3, 4], size=n),
        "spikes": lambda n: _add_spikes(np.zeros(n), rng),
        
        # Size variants
        "short_10": lambda: np.sin(np.linspace(0, 2*np.pi, 10)),
        "medium_50": lambda: np.sin(np.linspace(0, 4*np.pi, 50)),
        "large_500": lambda: np.sin(np.linspace(0, 8*np.pi, 500)),
    }
    
    return corpus
```

### Test Parameters

```python
PARITY_PARAMS = {
    "HVG": [
        {"method": "horizontal", "weighted": False, "limit": None},
        {"method": "horizontal", "weighted": True, "limit": None},
        {"method": "horizontal", "weighted": False, "limit": 10},
    ],
    "NVG": [
        {"method": "natural", "weighted": False, "limit": None},
        {"method": "natural", "weighted": True, "limit": None},
    ],
    "RN": [
        {"m": 2, "tau": 1, "rule": "epsilon", "epsilon": 0.5, "metric": "euclidean"},
        {"m": 3, "tau": 2, "rule": "knn", "k": 8, "metric": "euclidean"},
        {"m": 2, "tau": 1, "rule": "knn", "k": 5, "metric": "manhattan"},
        {"m": 2, "tau": 1, "rule": "knn", "k": 5, "metric": "maximum"},
    ],
    "TN": [
        {"method": "ordinal", "order": 3, "delay": 1},
        {"method": "ordinal", "order": 4, "delay": 1},
        {"method": "ordinal", "order": 3, "delay": 2},
    ],
}
```

## Equality Definitions

### For Small n (≤50): Edge Set Equality

```python
def check_edge_equality(G_r, G_p, tolerance=0):
    """Exact edge set comparison."""
    edges_r = set(frozenset(e) for e in G_r.edges())
    edges_p = set(frozenset(e) for e in G_p.edges())
    
    jaccard = len(edges_r & edges_p) / len(edges_r | edges_p)
    
    if tolerance == 0:
        return jaccard == 1.0
    else:
        return jaccard >= (1.0 - tolerance)
```

### For Large n (>50): Degree Distribution

```python
def check_degree_equality(G_r, G_p, tolerance=0.05):
    """Degree sequence L1 distance."""
    deg_r = sorted(dict(G_r.degree()).values())
    deg_p = sorted(dict(G_p.degree()).values())
    
    if len(deg_r) != len(deg_p):
        return False
    
    l1_dist = np.mean(np.abs(np.array(deg_r) - np.array(deg_p)))
    return l1_dist <= tolerance
```

### For Distance-Based Methods: Numeric Tolerance

```python
def check_adjacency_tolerance(A_r, A_p, rtol=1e-5, atol=1e-8):
    """Adjacency matrix with numeric tolerance."""
    return np.allclose(A_r, A_p, rtol=rtol, atol=atol)
```

### Summary Metrics (Always)

```python
def check_summary_metrics(G_r, G_p, tolerance=0.1):
    """Compare graph-level statistics."""
    metrics = {
        "n_nodes": (G_r.number_of_nodes(), G_p.number_of_nodes()),
        "n_edges": (G_r.number_of_edges(), G_p.number_of_edges()),
        "avg_degree": (np.mean([d for _, d in G_r.degree()]), 
                       np.mean([d for _, d in G_p.degree()])),
    }
    
    for name, (val_r, val_p) in metrics.items():
        rel_error = abs(val_r - val_p) / max(abs(val_r), 1e-10)
        if rel_error > tolerance:
            return False, name
    
    return True, None
```

## Acceptance Criteria

### Level 1: Strict Parity (Combinatorial Methods)

**Applies to:** HVG, NVG with limit=None, integer inputs

**Pass if:**
- Edge Jaccard = 1.0 (exact edge set match)
- Node count match
- All parameters explicitly specified (no defaults differ)

### Level 2: Numeric Parity (Distance-Based Methods)

**Applies to:** RN with fixed m/tau/metric, NVG with floats

**Pass if:**
- Edge Jaccard ≥ 0.95
- Degree L1 distance ≤ 0.1
- Summary metrics within 10% relative error

### Level 3: Distributional Parity (Large n)

**Applies to:** n > 100

**Pass if:**
- Degree distribution KS test p-value > 0.05
- Clustering coefficient within 20%
- Path length within 30%

### Default Behavior Test

```python
def test_default_parity():
    """Test with NO parameters passed - must match R defaults."""
    x = generate_sine_series(n=100, seed=42)
    
    # R: tsnet_vg(x)  # defaults: method="horizontal"
    # Python: HVG().fit_transform(x)
    
    # Must produce identical graphs
```

## Implementation Checklist

- [ ] Add `limit` parameter to HVG/NVG
- [ ] Implement or document embedding.dim auto-selection policy
- [ ] Add metric aliases (maximum→chebyshev)
- [ ] Implement quantiles symbolization
- [ ] Create `_validate_timeseries()` preprocessor
- [ ] Apply validation in all builders
- [ ] Generate test corpus with all edge cases
- [ ] Write parity harness with 3 tolerance levels
- [ ] Run full parity matrix (5 methods × 10 series × 3 param sets)
- [ ] Document all default values in API docs
- [ ] Add CI job that runs parity tests weekly

## Decisions

1. **Embedding auto-selection:** ✅ Implement FNN (False Nearest Neighbors)
2. **NaN handling:** ✅ Drop by default with logging, add `strict_validation` parameter
3. **Quantiles method:** ✅ R "quantiles" = Python "equal_freq" (confirmed)
4. **Node indexing:** ✅ Use 0-based (Python native), R graphs convert correctly
5. **igraph vs NetworkX:** ✅ GraphML preserves node IDs during conversion

