# ts2net Examples

This directory contains example scripts demonstrating ts2net functionality.

## Examples

### `quick_start.py`
Basic introduction to ts2net with synthetic data. Shows:
- HVG (Horizontal Visibility Graph)
- NetworkX conversion
- Performance mode
- Comparing different methods

**Run:**
```bash
python examples/quick_start.py
```

### `example_fred_data.py` ⭐ **NEW - Real Economic Data**
Comprehensive example using real economic data from FRED (Federal Reserve Economic Data). Demonstrates:
- Fetching correlated economic indicators (GDP, Unemployment, CPI)
- Univariate analysis with visibility graphs
- Multivariate network construction
- Recurrence and transition networks
- Network analysis and visualization

**Requirements:**
```bash
pip install pandas-datareader signalplot
```

**Note:** Uses `signalplot` for clean, minimalist time series plots. Falls back to matplotlib if signalplot is not available.

**Run:**
```bash
python examples/example_fred_data.py
```

**Output:**
Generated visualizations are saved to `examples/images/`:
- `fred_data.png` - Time series plots
- `fred_hvg_*.png` - Visibility graph networks
- `fred_proximity_network.png` - Proximity network from sliding windows

**Data Sources:**
- GDP (Gross Domestic Product)
- UNRATE (Unemployment Rate)
- CPIAUCSL (Consumer Price Index)

These series are naturally correlated and demonstrate how ts2net can reveal relationships between economic indicators.

### `example_r_parity.py`
Demonstrates R ts2net API parity for multivariate analysis:
- Multiple time series → network
- Distance functions (correlation, DTW, NMI, VOI, etc.)
- Network builders (k-NN, ε-NN, weighted)
- Window-based proximity networks

**Run:**
```bash
python examples/example_r_parity.py
```

### `viz_gallery.py` ⭐ **NEW - Visualization Gallery**
Demonstrates all five flagship visualization functions on the same dataset:
- **Figure 1**: Time series with change points and window boundaries
- **Figure 2**: Degree profile across time (local complexity proxy)
- **Figure 3**: Degree distribution as CCDF (log scale, stable comparison)
- **Figure 4**: Method comparison panel (edge count, avg degree, density)
- **Figure 5**: Window level feature map (heatmap for anomaly detection)
- **Bonus**: Small n graph drawing for HVG

**Run:**
```bash
python examples/viz_gallery.py
```

**Features:**
- Clean, scalable Matplotlib-only plots
- Consistent styling (no top/right spines, light gridlines)
- All functions return `(fig, ax)` for customization
- Designed for structure, stability, and comparison (not hairballs)

### `polars_spain_windows.py` ⭐ **NEW - Polars Ingestion & Windowing**
Demonstrates Polars-based Parquet ingestion and windowed analysis:
- Lazy-loading time series from Parquet using Polars
- Building window-level series per meter or region
- Running ts2net analysis with `output="stats"` for memory efficiency
- Writing results back to Parquet

**Requirements:**
```bash
pip install ts2net[polars]
# or
pip install polars pyarrow
```

**Run:**
```bash
python examples/polars_spain_windows.py
```

**Features:**
- Efficient lazy evaluation (only materializes what's needed)
- Time-based filtering and aggregation
- Windowed analysis for large time series
- Memory-efficient stats-only mode
- Outputs results to Parquet for downstream analysis

### `benchmark_numba.py`
Performance benchmarks comparing Numba-accelerated vs. pure Python implementations:
- HVG performance
- NVG performance
- Recurrence network performance
- Transition network performance

**Requirements:**
```bash
pip install numba  # For acceleration
```

**Run:**
```bash
python examples/benchmark_numba.py
```

## Installation

For examples with real data:
```bash
pip install ts2net[examples]
# or
pip install pandas-datareader
```

For performance benchmarks:
```bash
pip install ts2net[speed]
# or
pip install numba
```

## Notes

- All examples use logging for output (set level to INFO or DEBUG for more details)
- Some examples generate plots (saved as PNG files)
- FRED data example requires internet connection to fetch data
- Examples are designed to be educational and demonstrate best practices
