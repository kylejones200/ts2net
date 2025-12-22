"""
Example: Polars-based ingestion and windowing for Spanish energy data.

This example demonstrates:
- Loading time series from Parquet using Polars (lazy evaluation)
- Building window-level series per meter or region
- Running ts2net analysis with only_degrees=True for memory efficiency
- Writing results back to Parquet
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("⚠️  Polars not installed. Install with: pip install ts2net[polars]")
    sys.exit(1)

try:
    from ts2net.io_polars import load_series_from_parquet_polars
    from ts2net import HVG, graph_summary
    from ts2net.core import graph_summary as core_graph_summary
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_series_windowed(
    series_dict: dict[str, np.ndarray],
    window_width: int = 24,
    step: int = 1,
    method: str = "hvg",
    only_degrees: bool = True
):
    """
    Analyze time series with sliding windows using ts2net.
    
    Parameters
    ----------
    series_dict : dict[str, np.ndarray]
        Dictionary mapping series ID to values array
    window_width : int
        Width of sliding window
    step : int
        Step size for sliding window
    method : str
        Network method: 'hvg', 'nvg', 'recurrence', 'transition'
    only_degrees : bool
        If True, only compute degrees (memory efficient)
    
    Returns
    -------
    dict
        Results per series ID
    """
    from ts2net import build_network
    
    results = {}
    
    for series_id, values in series_dict.items():
        logger.info(f"Processing {series_id} ({len(values)} points)...")
        
        # Create sliding windows
        n_windows = (len(values) - window_width) // step + 1
        if n_windows <= 0:
            logger.warning(f"  Series {series_id} too short for window_width={window_width}")
            continue
        
        window_stats = []
        
        for i in range(0, len(values) - window_width + 1, step):
            window = values[i:i + window_width]
            
            # Skip windows with all NaN or constant values
            if np.all(np.isnan(window)) or np.std(window) == 0:
                continue
            
            # Build network for this window
            try:
                if method == 'recurrence':
                    from ts2net import RecurrenceNetwork
                    builder = RecurrenceNetwork(m=3, rule='knn', k=5, only_degrees=only_degrees)
                    g = builder.build(window)
                elif method == 'transition':
                    from ts2net import TransitionNetwork
                    builder = TransitionNetwork(symbolizer='ordinal', order=3, only_degrees=only_degrees)
                    g = builder.build(window)
                elif method == 'hvg':
                    from ts2net import HVG
                    builder = HVG(only_degrees=only_degrees)
                    g = builder.build(window)
                elif method == 'nvg':
                    from ts2net import NVG
                    builder = NVG(only_degrees=only_degrees)
                    g = builder.build(window)
                else:
                    g = build_network(window, method, only_degrees=only_degrees)
                
                # Get statistics
                degrees = g.degree_sequence()
                stats = {
                    'window_start': i,
                    'window_end': i + window_width,
                    'n_nodes': g.n_nodes,
                    'n_edges': g.n_edges,
                    'deg_mean': float(np.mean(degrees)),
                    'deg_std': float(np.std(degrees)),
                    'deg_min': int(np.min(degrees)),
                    'deg_max': int(np.max(degrees)),
                }
                
                window_stats.append(stats)
                
            except Exception as e:
                logger.warning(f"  Error processing window {i}: {e}")
                continue
        
        results[series_id] = window_stats
        logger.info(f"  Processed {len(window_stats)} windows")
    
    return results


def write_results_to_parquet(results: dict, output_path: str):
    """Write analysis results to Parquet file."""
    # Flatten results into rows
    rows = []
    for series_id, window_stats in results.items():
        for stat in window_stats:
            row = {'series_id': series_id, **stat}
            rows.append(row)
    
    if not rows:
        logger.warning("No results to write")
        return
    
    # Create Polars DataFrame and write
    df = pl.DataFrame(rows)
    df.write_parquet(output_path)
    logger.info(f"✅ Results written to {output_path}")
    logger.info(f"   {len(rows)} rows, {len(results)} series")


def main():
    """Main example function."""
    logger.info("=" * 60)
    logger.info("Polars-based Time Series Analysis")
    logger.info("=" * 60)
    logger.info("")
    
    # Example: Load Spanish energy consumption data
    # This is a template - replace with your actual Parquet file path
    parquet_path = "data/spain_energy.parquet"  # Replace with actual path
    
    logger.info("📊 Loading time series from Parquet...")
    logger.info(f"   Path: {parquet_path}")
    
    # Check if file exists (for demo purposes)
    if not Path(parquet_path).exists():
        logger.warning(f"⚠️  File not found: {parquet_path}")
        logger.info("")
        logger.info("This is a template example. To use:")
        logger.info("1. Prepare a Parquet file with columns: timestamp, consumption, meter_id")
        logger.info("2. Update parquet_path in this script")
        logger.info("3. Run the script")
        logger.info("")
        logger.info("Example Parquet schema:")
        logger.info("  - timestamp: datetime")
        logger.info("  - consumption: float64")
        logger.info("  - meter_id: string (optional)")
        return
    
    try:
        # Load series by meter_id with hourly aggregation
        series = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='consumption',
            id_col='meter_id',  # Group by meter
            freq='1h',  # Hourly aggregation
            agg='mean',
            tz='Europe/Madrid',  # Spanish timezone
        )
        
        logger.info(f"✅ Loaded {len(series)} series")
        logger.info(f"   Series IDs: {list(series.keys())[:5]}...")
        logger.info("")
        
        # Analyze with sliding windows
        logger.info("🔍 Analyzing with sliding windows (24-hour windows)...")
        results = analyze_series_windowed(
            series,
            window_width=24,  # 24-hour windows
            step=1,  # 1-hour step
            method='hvg',
            only_degrees=True  # Memory efficient
        )
        
        # Write results
        output_path = "results/spain_energy_analysis.parquet"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        write_results_to_parquet(results, output_path)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Analysis complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
