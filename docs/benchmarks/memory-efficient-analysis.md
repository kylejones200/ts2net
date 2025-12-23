# Memory-Efficient Time Series Network Analysis

## What Problem This Solves

Traditional time series network construction methods (HVG, NVG, Recurrence Networks) hit memory and computational walls at scale. The standard approach builds dense adjacency matrices that require O(n²) memory, causing Out-Of-Memory (OOM) failures for datasets with more than ~10,000 points.

**Specific problems solved:**
- **Memory explosion**: Dense n×n adjacency matrices require 63 GB for 88,890 nodes (float64)
- **Computational bottlenecks**: O(n²) algorithms that don't leverage theoretical optimizations
- **Scale limitations**: Unable to process real-world smart meter data (50k-100k points per meter)
- **Multi-series analysis**: Can't analyze hundreds of thousands of time series in one run

## What Scale It Works At

**Single Time Series:**
- 88,890 points (Morocco dataset) - 2.5 minutes, <50 MB RAM
- 2.3M points (Spain 50 meters) - 42 seconds
- Theoretical limit: Millions of points with O(n) HVG and bounded NVG

**Multi-Series Analysis:**
- 283,690 time series (ELEC dataset) - 9 minutes total
- 43.4 million total data points analyzed
- Processing rate: 524 series/second sustained

**Memory footprint:**
- Dense matrix approach: 63 GB for 88k points
- Our approach: <50 MB for 88k points
- **~1,260x memory reduction**

## What It Replaces

**Old approach (causing failures):**
```python
# Build full dense adjacency matrix
A = np.zeros((n_nodes, n_nodes))  # OOM at n > 10k
for edge in edges:
    A[i, j] = 1
# Compute metrics from dense matrix
degrees = np.sum(A, axis=1)
```

**New approach (memory-efficient):**
```python
# Option 1: Compute degrees directly from edges
degrees = np.zeros(n_nodes, dtype=np.int64)
for edge in edges:
    degrees[i] += 1
    degrees[j] += 1

# Option 2: Use Rust to compute and return only degrees
degrees_hvg = hvg_degrees(x)  # Returns degree vector, not edges
degrees_nvg = nvg_degrees_sweepline(x, limit=2000)
```

**Key architectural changes:**
1. **No dense matrices** - use sparse edge lists or degree-only computation
2. **Stream processing** - process edges in chunks, never materialize full graph
3. **Algorithmic optimization** - O(n) stack-based HVG instead of naive O(n²)
4. **Horizon limits** - bound NVG visibility distance to control memory
5. **Rust acceleration** - compute-intensive parts in Rust, return minimal data to Python

## Results: Morocco Smart Meter Analysis

![Morocco Full Resolution Network Analysis](morocco_full_resolution.png)

**Dataset:** 88,890 hourly readings (10+ years) across 9 zones

**Performance:**
- Runtime: 2.5 minutes (vs 6+ hours timeout with old approach)
- Memory: <50 MB peak (vs 63 GB+ with dense matrices)
- All methods: HVG, NVG, Transition, Recurrence Networks

**Key findings:**
- HVG produces ~180k edges (avg degree 3.96, theoretical 4.0)
- NVG shows higher connectivity (avg degree 11.7)
- Network complexity reveals consumption patterns
- Different zones show distinct network signatures

**Network properties captured:**
- Temporal structure through visibility graphs
- Symbolic dynamics through transition networks  
- Recurrence patterns through phase space analysis
- All at full temporal resolution (no downsampling required)

