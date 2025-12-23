# Large-Scale Benchmark Results

This document summarizes large-scale experiments demonstrating ts2net's capability to handle real-world time series datasets at scale.

## Overview

These experiments validate ts2net's performance and memory efficiency on datasets ranging from single large time series (88k+ points) to massive multi-series analysis (283k+ series, 43M+ data points).

Key Achievements:
- Zero Out-of-Memory failures across all experiments
- Predictable linear scaling with series length
- Memory footprint reduced by 1,260x vs. dense matrix approaches
- Validated theoretical properties (HVG degree ≈ 4.0) at unprecedented scale

## Experiments

### 1. Morocco Smart Meter Full Resolution
Location: `experiments/morocco-full-resolution/`

Dataset:
- 88,890 hourly readings per zone (10+ years)
- 9 geographic zones analyzed
- Full temporal resolution (no downsampling)

Results:
- Runtime: 2.5 minutes total
- Memory: <50 MB peak
- HVG average degree: 3.999 (validates theoretical 4.0)
- All methods successful: HVG, NVG, Transition Networks, Recurrence Networks

Key Finding: HVG degree theory holds perfectly even at 88k-point scale, validating the algorithm's correctness.

Files:
- `README.md` - Complete experiment documentation
- `morocco_full_resolution.csv` - Per-zone network metrics
- `morocco_full_resolution.png` - Multi-panel visualization
- `analyze_morocco_full.py` - Reproducible analysis script

---

### 2. Spain Smart Meter Multi-Series
Location: `experiments/spain-multi-meter/`

Dataset:
- 633M rows in source dataset
- 50 meters analyzed (top meters by reading count)
- 2.3M hourly readings total
- Temporal resolution: Resampled from 30-min to 1-hour

Results:
- Runtime: 42 seconds for 50 meters
- Processing rate: ~1.2 meters/second
- HVG degree: 3.99 consistently across all meters
- NVG variability: 4.95 to 22.17 (reveals consumption pattern diversity)

Key Finding: Wide NVG degree variability (4.95-22.17) reveals meter heterogeneity, enabling consumption pattern classification.

Files:
- `README.md` - Complete experiment documentation
- `spain_meter_network_results.csv` - Per-meter metrics (50 rows × 12 columns)
- `spain_meter_network_analysis.png` - Multi-panel visualization
- `analyze_spain.py` - Reproducible analysis script

---

### 3. ELEC Full Dataset (Extreme Scale)
Location: `experiments/elec-full-dataset/`

Dataset:
- 593,134 time series in source (EIA US Electricity Data)
- 283,690 valid series processed (47.8% coverage)
- 43.4 million total data points
- Series length: 50 to 295 months (median: 98)

Results:
- Runtime: 9 minutes total
- Memory: <200 MB peak (constant per series)
- Processing rate: 524 series/second sustained
- Zero OOM errors

Key Findings:
1. HVG degree distribution peaks sharply at 4.0 for longer series, validating theory across 283k diverse electricity metrics
2. Dataset contains varied measurements: power generation (wind, solar, natural gas, coal, nuclear), fuel consumption, state-level and plant-level aggregates
3. Geographic coverage spans all US states (California dominates with 22k+ series)

Files:
- `README.md` - Complete experiment documentation
- `elec_full_network_results.csv` - Per-series metrics (283,690 rows × 15 columns, 73 MB)
- `elec_full_summary_stats.csv` - Aggregate statistics
- `elec_full_network_analysis.png` - 12-panel visualization
- `analyze_elec_full.py` - Streaming analysis script

---

## Performance Summary

| Experiment | Series Count | Total Points | Runtime | Peak Memory | Rate |
|------------|--------------|--------------|---------|-------------|------|
| Morocco | 9 zones | 800k | 2.5 min | <50 MB | - |
| Spain | 50 meters | 2.3M | 42 sec | <100 MB | 1.2 series/sec |
| ELEC | 283,690 | 43.4M | 9 min | <200 MB | 524 series/sec |

### Method Performance

| Method | Single Series (88k) | Multi-Series Rate | Notes |
|--------|---------------------|-------------------|-------|
| HVG | ~1 second | Very fast | O(n) algorithm, validated theory |
| NVG | ~5 seconds | Fast | Requires `limit` parameter for large n |
| Transition | <1 second | Very fast | Symbolic dynamics |
| Recurrence | ~30 sec (resampled) | Too slow | Excluded from multi-series at scale |

---

## Memory Efficiency

### Before Optimization
- Dense adjacency matrices: 63 GB required for 88,890 nodes (float64)
- OOM failures: Common for n > 10,000 points
- Limited scale: Multi-series analysis restricted to <50 series

### After Optimization
- Sparse edge lists: <50 MB for 88k points
- Degree-only computation: Even more memory-efficient
- Zero OOM failures: All experiments completed successfully

Improvement: 1,260x memory reduction (63 GB → 50 MB)

### Memory Estimates

| Approach | Memory (88k nodes) | Notes |
|----------|-------------------|-------|
| Dense adjacency | ~63 GB | 8 bytes × n² |
| Sparse adjacency | ~1.6 MB | 16 bytes × m (100k edges) |
| Edge list | ~1.6 MB | 16 bytes × m |
| Degrees only | ~720 KB | 8 bytes × n |

---

## Validation Results

### HVG Theoretical Property Validation

The Horizontal Visibility Graph (HVG) has a well-known theoretical property: for typical time series, the average degree converges to 4.0 as series length increases.

Validation across experiments:
- Morocco (88k points): Average degree = 3.999 across 9 zones
- Spain (2.3M points): Average degree = 3.99 across 50 meters  
- ELEC (283k series): Degree distribution peaks at 4.0 for longer series

This validates both:
1. The correctness of ts2net's HVG implementation
2. The theoretical property holds at unprecedented scale

### Linear Scaling Validation

HVG edge count scales linearly with series length: `edges ≈ 2n`

Observed across all experiments:
- Perfect linear relationship in edge vs. length plots
- Confirms O(n) algorithm complexity
- No quadratic blowup even at 88k+ points

---

## Technical Documentation

For detailed explanations of the optimizations and algorithms:

- Memory Efficiency: `docs/benchmarks/memory-efficient-analysis.md`
- Rust Acceleration: `docs/benchmarks/rust-degree-computation.md`
- Horizon-Limited NVG: `docs/benchmarks/horizon-limited-nvg.md`

---

## Reproducibility

All experiments include:
- Complete Python scripts with configuration
- Expected runtimes and outputs documented
- CSV results files with all metrics
- Visualization PNG files
- Detailed README in each experiment folder

### Running Experiments

```bash
# Morocco full resolution
cd experiments/morocco-full-resolution
python analyze_morocco_full.py

# Spain multi-meter
cd experiments/spain-multi-meter
python analyze_spain.py

# ELEC full dataset (requires large CSV input)
cd experiments/elec-full-dataset
python analyze_elec_full.py
```

Note: Some experiments require access to specific datasets. See each experiment's README for data source information.

---

## Impact

These benchmarks demonstrate that ts2net can handle:
1. Real-world smart meter data at full temporal resolution
2. Large-scale comparative studies (thousands of series)
3. Production deployments with predictable memory usage
4. Research-scale analyses (hundreds of thousands of series)

Before: Limited to <10k points, frequent OOM failures, unpredictable performance

After: Scales to 88k+ points, zero failures, predictable linear scaling


---

Last Updated: December 2025  
Location: `experiments/` directory and `docs/benchmarks/`

