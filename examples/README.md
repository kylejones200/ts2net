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
