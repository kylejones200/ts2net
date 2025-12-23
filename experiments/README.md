# Large-Scale Experiment Results

This directory contains evidence from large-scale experiments demonstrating ts2net's scalability and performance on real-world datasets.

## Overview

These experiments validate ts2net's capability to handle:
- Single large time series (88k+ points)
- Multi-series analysis (50+ series)
- Extreme-scale batch processing (283k+ series, 43M+ data points)

All experiments achieved:
- Zero Out-of-Memory failures
- Predictable linear scaling
- Memory efficiency (1,260x reduction vs. dense matrices)
- Validation of theoretical properties at scale

## Experiments

### 1. Morocco Full Resolution
**Directory:** `morocco-full-resolution/`

Single large time series analysis (88,890 points × 9 zones).

**Key Results:**
- 2.5 minutes runtime
- <50 MB peak memory
- HVG degree = 3.999 (validates theoretical 4.0)

### 2. Spain Multi-Meter
**Directory:** `spain-multi-meter/`

Multi-series analysis (50 meters, 2.3M total points).

**Key Results:**
- 42 seconds runtime
- NVG degree variability: 4.95 to 22.17 (reveals consumption pattern diversity)

### 3. ELEC Full Dataset
**Directory:** `elec-full-dataset/`

Extreme-scale batch processing (283,690 series, 43.4M points).

**Key Results:**
- 9 minutes runtime
- 524 series/second processing rate
- <200 MB constant memory footprint

## File Structure

Each experiment directory contains:
- `README.md` - Complete experiment documentation (configuration, results, findings)
- `*.csv` - Results data (network metrics per series/zone/meter)
- `*.png` - Visualizations (multi-panel figures)
- `analyze_*.py` - Analysis scripts (reference implementations)

## Using These Results

### For Evidence/Publications

These results can be cited as evidence of:
- ts2net's scalability to real-world datasets
- Memory efficiency improvements over dense matrix approaches
- Validation of theoretical properties (HVG degree ≈ 4.0) at scale
- Practical performance benchmarks

See `BENCHMARKS.md` in the repository root for a comprehensive summary.

### For Reproducing Experiments

**Note:** These scripts were run in a specific experimental environment. To reproduce:

1. Install ts2net with all dependencies
2. Obtain the datasets (see each experiment's README for dataset information)
3. Update data paths in the scripts as needed
4. Run the analysis scripts

The scripts serve as reference implementations showing the exact analysis performed.

### For Understanding Scale Capabilities

Each experiment demonstrates different aspects of scalability:
- **Morocco:** Single large series, all methods
- **Spain:** Multi-series, memory-efficient processing
- **ELEC:** Extreme-scale batch processing with streaming

## Summary Statistics

| Experiment | Series | Points | Runtime | Memory | Key Metric |
|------------|--------|--------|---------|--------|------------|
| Morocco | 9 zones | 800k | 2.5 min | <50 MB | HVG deg = 3.999 |
| Spain | 50 meters | 2.3M | 42 sec | <100 MB | NVG deg range: 4.95-22.17 |
| ELEC | 283,690 | 43.4M | 9 min | <200 MB | 524 series/sec |

## Related Documentation

- **`BENCHMARKS.md`** (repository root) - Comprehensive benchmark summary
- **`docs/benchmarks/`** - Technical documentation on optimizations:
  - `memory-efficient-analysis.md` - Core memory optimization strategy
  - `rust-degree-computation.md` - Rust acceleration details
  - `horizon-limited-nvg.md` - Bounded NVG algorithm

---

**Purpose:** Evidence of ts2net's scalability and performance on real-world datasets  
**Last Updated:** December 2025

