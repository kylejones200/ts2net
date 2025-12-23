# Rust-Accelerated Degree-Only Computation

## What Problem This Solves

Even with sparse edge list representations, transferring millions of edges from Rust to Python creates a bottleneck:
- **Memory transfer overhead**: Large edge arrays (2×n entries) cross the Rust/Python boundary
- **Python processing cost**: Computing degrees from edge lists in Python is slow
- **Unnecessary data**: Most network analyses only need degree sequences, not full edge lists

**Example bottleneck:**
For a series with 88,890 points, HVG generates ~180k edges:
- Edge array: 360k integers = 2.8 MB to transfer
- Degree vector: 88k integers = 0.7 MB (4x smaller)
- If you only need degrees, why transfer edges at all?

## What Scale It Works At

**Memory transfer reduction:**
- **Before**: Return 2n integers (edge pairs)
- **After**: Return n integers (degrees only)
- **Savings**: 50% memory transfer for each network method

**Processing speedup:**
Computing degrees in Rust vs Python:
- **Rust**: Direct increment during edge generation (zero extra cost)
- **Python**: Loop through edge list, update dictionary (significant overhead)

**Tested at:**
- 88,890 points: Degree computation negligible vs edge generation
- 283,690 series: Consistent per-series overhead reduction
- Works for HVG, NVG (with horizon limits)

## What It Replaces

**Old API (returns edges):**
```rust
// In ts2net_rs/src/graphs/visibility.rs
#[pyfunction]
pub fn hvg_edges(y: &PyArray1<f64>) -> Vec<(usize, usize)> {
    // Generate edges
    // Return full edge list to Python
    edges
}
```

```python
# In Python
edges = hvg_edges(time_series)  # Large transfer
degrees = compute_degrees_from_edges(edges, n)  # Python loop
```

**New API (returns degrees directly):**
```rust
// In ts2net_rs/src/graphs/visibility.rs
#[pyfunction]
pub fn hvg_degrees(y: &PyArray1<f64>) -> Array1<usize> {
    let n = y.len();
    let mut degrees = Array1::<usize>::zeros(n);
    let mut stack = Vec::with_capacity(n);
    
    for j in 0..n {
        while let Some(&i) = stack.last() {
            if y[i] < y[j] {
                stack.pop();
                degrees[i] += 1;  // Update during generation
                degrees[j] += 1;
            } else {
                break;
            }
        }
        if let Some(&i) = stack.last() {
            degrees[i] += 1;
            degrees[j] += 1;
        }
        stack.push(j);
    }
    degrees
}
```

```python
# In Python  
degrees = hvg_degrees(time_series)  # Small transfer, no post-processing
```

## Implementation Details

**Added to `ts2net_rs/src/graphs/visibility.rs`:**

1. **`hvg_degrees(y: &Array1<f64>) -> Array1<usize>`**
   - Stack-based O(n) algorithm
   - Increments degree counters during edge discovery
   - Returns degree vector only

2. **`nvg_degrees_sweepline(y: &Array1<f64>, limit: Option<usize>) -> Array1<usize>`**
   - Sweepline algorithm with optional horizon limit
   - Bounds edges per node when `limit` is set
   - Memory-safe for long series

**Exposed to Python in `ts2net_rs/src/lib.rs`:**
```rust
m.add_function(wrap_pyfunction!(hvg_degrees, m)?)?;
m.add_function(wrap_pyfunction!(nvg_degrees_sweepline, m)?)?;
```

**Usage pattern:**
```python
from ts2net_rs import hvg_degrees, nvg_degrees_sweepline

# Compute degrees directly
degrees_hvg = hvg_degrees(time_series)
degrees_nvg = nvg_degrees_sweepline(time_series, limit=2000)

# Compute metrics
avg_degree = np.mean(degrees_hvg)
max_degree = np.max(degrees_hvg)
degree_distribution = np.bincount(degrees_hvg)
```

## Benefits

1. **Reduced memory transfer**: 50% less data crossing Rust/Python boundary
2. **Faster processing**: Degree computation happens during edge generation (free)
3. **Cleaner API**: One function call instead of two steps
4. **Memory safety**: Degree array is always O(n), never grows unexpectedly
5. **Composability**: Can still call edge-returning functions when full graph is needed

## When to Use

**Use degree-only functions when:**
- Computing summary statistics (mean, max, distribution)
- Comparing network complexity across series
- You don't need edge weights or specific connectivity structure
- Processing many series (multi-series analysis)

**Use edge-returning functions when:**
- Need to save full graph structure
- Computing shortest paths or centralities requiring graph traversal
- Building explicit network for visualization
- Analyzing specific edge properties

## Figure: Memory Transfer Comparison

This optimization is invisible to end users but critical for performance. The Morocco and ELEC experiments benefited from this approach by reducing memory pressure during processing.

**Conceptual improvement:**
- Dense matrix: O(n²) memory (not recommended)
- Edge list: O(edges) memory (efficient)
- Degree vector: O(n) memory (smallest possible)

For HVG where edges ≈ 2n, degree-only saves ~50% transfer.
For NVG where edges can be much larger, savings are even more significant.

