# Horizon-Limited Natural Visibility Graphs

## What Problem This Solves

Natural Visibility Graphs (NVG) can generate excessive edges for smooth or slowly-varying time series. Without bounds, a single peak can be "visible" to thousands of distant points, leading to:

1. **Memory explosion**: Edge list grows to millions of entries
2. **Loss of locality**: Long-range connections dilute meaningful local structure  
3. **Computational cost**: O(n²) in worst case for smooth series
4. **Interpretation difficulty**: Dense graphs lose discriminative power

**Example problem:**
For a smooth sine wave of length 88,890:
- Unbounded NVG: ~520k edges (avg degree 11.7)
- With limit=2000: Similar local structure, bounded memory

## What Scale It Works At

**Without horizon limit:**
- Works for series up to ~100k points if edges stay reasonable
- Fails for smooth series where edges grow quadratically
- Memory unpredictable (depends on data characteristics)

**With horizon limit (e.g., 2000):**
- Works for series of any length (memory now O(n×limit))
- Bounded worst case: 2000 edges per node maximum
- Preserves local visibility structure
- Predictable memory usage

**Tested at:**
- Spain dataset: 50 meters × ~47k points with limit=2000 (no issues)
- Morocco dataset: 88k points, natural limit from data properties
- Theoretical: Could handle millions of points with limit=2000

## What It Replaces

**Old approach (unbounded NVG):**
```rust
pub fn nvg_edges_sweepline(y: &Array1<f64>) -> Vec<(usize, usize)> {
    let n = y.len();
    let mut edges = Vec::new();
    
    for i in 0..n {
        let yi = y[i];
        let mut slope_max = f64::NEG_INFINITY;
        
        // Check ALL points to the right
        for j in (i + 1)..n {  // Unbounded: can be huge
            let slope = (y[j] - yi) / ((j - i) as f64);
            if slope > slope_max {
                edges.push((i, j));
                slope_max = slope;
            }
        }
    }
    edges
}
```

**New approach (horizon-limited NVG):**
```rust
pub fn nvg_degrees_sweepline(y: &Array1<f64>, limit: Option<usize>) -> Array1<usize> {
    let n = y.len();
    let mut degrees = Array1::<usize>::zeros(n);
    let actual_limit = limit.unwrap_or(n);  // Default to n if no limit
    
    for i in 0..n {
        let yi = y[i];
        let mut slope_max = f64::NEG_INFINITY;
        let end_j = (i + actual_limit + 1).min(n);  // Bound the search
        
        for j in (i + 1)..end_j {  // Limited range
            let slope = (y[j] - yi) / ((j - i) as f64);
            if slope > slope_max {
                degrees[i] += 1;
                degrees[j] += 1;
                slope_max = slope;
            }
        }
    }
    degrees
}
```

## Implementation

**Function signature:**
```rust
pub fn nvg_degrees_sweepline(
    y: &Array1<f64>,      // Time series
    limit: Option<usize>  // Optional horizon limit
) -> Array1<usize>        // Degree sequence
```

**Usage:**
```python
from ts2net_rs import nvg_degrees_sweepline

# Unbounded (original behavior)
degrees = nvg_degrees_sweepline(time_series, None)

# Limited to 2000 time steps ahead
degrees = nvg_degrees_sweepline(time_series, limit=2000)
```

**Choosing the limit:**
- **No limit** (`None`): Use for short series (<10k points) or when you need exact NVG
- **limit=500**: Captures sub-daily patterns in hourly data
- **limit=2000**: Good for daily patterns in hourly data (83 days)
- **limit=8760**: One year of hourly data
- **limit=n/10**: Conservative choice for any series length

## Complexity Analysis

**Time complexity:**
- Unbounded: O(n²) worst case (smooth series)
- With limit=L: O(n×L) = O(n) when L is constant

**Space complexity:**
- Unbounded: O(edges) which can be O(n²)
- With limit=L: O(n×L) edges maximum, or just O(n) if returning degrees only

**Memory guarantee with limit=2000:**
- Maximum edges: 2000×n
- For n=88,890: at most 177M edges (but degree computation avoids storing them)
- Degree vector: always just 88,890 integers

## Trade-offs

**What you keep:**
- Local visibility structure (most important for pattern detection)
- Temporal causality (only look forward in time)
- Efficiency gains over HVG (NVG still finds more edges than HVG in local windows)

**What you lose:**
- Very long-range connections (but these are often spurious for noisy data)
- Exact NVG definition (now it's "local NVG")

**Why it's worth it:**
- Predictable memory and runtime
- Preserves discriminative power for most time series analysis tasks
- Scales to arbitrarily long series
- Interpretability: "visibility within horizon" is more meaningful than unbounded

## Validation

**Spain experiment:**
- 50 meters processed with limit=2000
- Average NVG degree: 11.90 (reasonable, not explosive)
- No memory issues despite 47k+ points per meter
- Runtime: 42 seconds total

**Comparison (conceptual):**
- Unbounded NVG on smooth 100k series: could generate 5M+ edges (fails)
- Limited NVG with limit=2000: generates <200M edge checks = manageable

## Recommended Usage

```python
# For hourly smart meter data
degrees = nvg_degrees_sweepline(hourly_series, limit=2000)  # ~83 days

# For daily financial data
degrees = nvg_degrees_sweepline(daily_series, limit=252)  # ~1 trading year

# For high-frequency data (minutely)
degrees = nvg_degrees_sweepline(minute_series, limit=1440)  # 1 day

# Short series (<5k points) - no limit needed
degrees = nvg_degrees_sweepline(short_series, None)
```

## Figure: Scaling Comparison

See Spain and ELEC experiments for validation that limited NVG works at scale without sacrificing analytical power. The key insight: local structure matters more than distant visibility for most time series analysis tasks.

