#!/usr/bin/env python3
"""
YAML-based pipeline runner for ts2net.

Loads configuration from YAML, validates it, and runs the analysis pipeline.
Keeps "what" in YAML, "how" in Python.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from ts2net.io_polars import load_series_from_parquet_polars
    from ts2net.config import PipelineConfig
    from ts2net.factory import build_graph_from_config
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_series(config: PipelineConfig) -> Dict[str, np.ndarray]:
    """Load time series from dataset configuration."""
    dataset = config.dataset
    path = dataset.path
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Use Polars if available, otherwise fallback to pandas
    if HAS_POLARS:
        logger.info(f"Loading series from {path} using Polars...")
        series = load_series_from_parquet_polars(
            path=path,
            time_col=dataset.time_col,
            value_col=dataset.value_col,
            id_col=dataset.id_col,
            start=dataset.start,
            end=dataset.end,
            freq=config.sampling.frequency,
            agg=config.sampling.agg,
            tz=dataset.tz
        )
    else:
        # Fallback to pandas (would need to implement)
        raise ImportError("Polars required for data loading. Install with: pip install ts2net[polars]")
    
    logger.info(f"Loaded {len(series)} series")
    return series


# build_graph function removed - using factory.build_graph_from_config instead


def process_series(series_id: str, series_data: np.ndarray, config: PipelineConfig) -> Dict[str, Any]:
    """Process a single series with all enabled graph types."""
    result = {'series_id': series_id, 'n_points': len(series_data)}
    
    # Handle BSTS decomposition if enabled
    bsts_config = config.bsts
    series_to_analyze = series_data
    bsts_enabled = False
    
    if bsts_config.enabled:
        try:
            from ts2net.bsts import features, BSTSSpec
            
            # Build BSTS spec from config
            bsts_spec = BSTSSpec(
                level=bsts_config.level,
                trend=bsts_config.trend,
                seasonal_periods=bsts_config.seasonal_periods,
                robust=bsts_config.robust,
                standardize_residual=bsts_config.standardize_residual
            )
            
            # Determine window mode if series is too long
            window = bsts_config.window
            if window is None and len(series_data) > bsts_config.max_points:
                window = bsts_config.max_points  # Auto-enable windowing for long series
            
            # Get enabled graph methods
            graphs = config.graphs
            enabled_methods = [
                'hvg' if graphs.hvg.enabled else None,
                'nvg' if graphs.nvg.enabled else None,
                'recurrence' if graphs.recurrence.enabled else None,
                'transition' if graphs.transition.enabled else None,
            ]
            enabled_methods = [m for m in enabled_methods if m is not None]
            if not enabled_methods:
                enabled_methods = ['hvg', 'transition']  # Default
            
            # Extract features (includes decomposition + residual analysis)
            feat_result = features(
                series_data,
                methods=enabled_methods,
                bsts=bsts_spec,
                window=window,
                nvg_limit=graphs.nvg.limit if graphs.nvg.limit else 3000
            )
            
            # Add structural and residual stats to result
            result['raw_stats'] = feat_result.raw_stats
            result['structural_stats'] = feat_result.structural_stats
            result['residual_network_stats'] = feat_result.residual_network_stats
            
            bsts_enabled = True
            
        except ImportError:
            logger.warning(f"{series_id}: statsmodels not available, skipping BSTS")
        except Exception as e:
            logger.warning(f"{series_id} BSTS: {e}")
            if config.logging.log_errors:
                result['bsts_error'] = str(e)
    
    # Standard graph analysis (only if BSTS not enabled)
    # If BSTS is enabled, residual network stats are already computed
    if not bsts_enabled:
        graphs = config.graphs
        
        # Dispatch dictionary for graph processing
        graph_configs = {
            'hvg': (graphs.hvg, graphs.hvg.enabled),
            'nvg': (graphs.nvg, graphs.nvg.enabled),
            'recurrence': (graphs.recurrence, graphs.recurrence.enabled),
            'transition': (graphs.transition, graphs.transition.enabled),
        }
        
        for graph_type, (graph_config, enabled) in graph_configs.items():
            if not enabled:
                continue
            
            try:
                stats = build_graph_from_config(
                    series_to_analyze,
                    graph_type,
                    graph_config
                )
                result[graph_type] = stats
            except Exception as e:
                logger.warning(f"{series_id} {graph_type}: {e}")
                if config.logging.log_errors:
                    result[f'{graph_type}_error'] = str(e)
    
    return result


def process_windowed(series_id: str, series_data: np.ndarray, config: PipelineConfig) -> list:
    """Process series with sliding windows."""
    windows_config = config.windows
    window_size = windows_config.size
    step = windows_config.step if windows_config.step else 1
    
    if window_size is None:
        # Process full series
        return [process_series(series_id, series_data, config)]
    
    results = []
    n_windows = (len(series_data) - window_size) // step + 1
    
    for i in range(0, len(series_data) - window_size + 1, step):
        window = series_data[i:i + window_size]
        
        # Skip invalid windows
        if np.all(np.isnan(window)) or np.std(window) == 0:
            continue
        
        window_result = process_series(
            f"{series_id}_w{i}",
            window,
            config
        )
        window_result['window_start'] = i
        window_result['window_end'] = i + window_size
        results.append(window_result)
    
    return results


def write_output(results: list, config: PipelineConfig):
    """Write results to output file."""
    output_config = config.output
    output_path = output_config.path
    output_format = output_config.format
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Dispatch dictionary for output formats
    output_writers = {
        'parquet': lambda: _write_parquet(results, output_path),
        'json': lambda: _write_json(results, output_path),
        'csv': lambda: _write_csv(results, output_path),
    }
    
    writer = output_writers.get(output_format.lower())
    if writer is None:
        raise ValueError(f"Unknown output format: {output_format}. Must be one of {list(output_writers.keys())}")
    
    writer()
    logger.info(f"Results written to {output_path}: {len(results)} rows")


def _write_parquet(results: list, output_path: str):
    """Write results to Parquet format."""
    if not HAS_POLARS:
        raise ImportError("Polars required for Parquet output")
    df = pl.DataFrame(results)
    df.write_parquet(output_path)


def _write_json(results: list, output_path: str):
    """Write results to JSON format."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def _write_csv(results: list, output_path: str):
    """Write results to CSV format."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def write_errors(errors: list, config: PipelineConfig):
    """Write error log if configured."""
    logging_config = config.logging
    if logging_config.log_errors and logging_config.error_path:
        with open(logging_config.error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.info(f"Error log written to {logging_config.error_path}")


def main(config_path: str):
    """Main pipeline execution."""
    logger.info(f"Loading configuration from {config_path}")
    config = PipelineConfig.from_yaml(config_path)
    
    dataset_name = config.dataset.name
    logger.info(f"Dataset: {dataset_name}")
    
    # Load series
    series_dict = load_series(config)
    
    # Process each series
    all_results = []
    errors = []
    
    windows_enabled = config.windows.enabled
    
    for series_id, series_data in series_dict.items():
        try:
            if windows_enabled:
                results = process_windowed(series_id, series_data, config)
            else:
                results = [process_series(series_id, series_data, config)]
            
            all_results.extend(results)
            
        except Exception as e:
            error_msg = f"{series_id}: {e}"
            logger.error(error_msg)
            errors.append({'series_id': series_id, 'error': str(e)})
    
    # Write output
    if all_results:
        write_output(all_results, config)
    else:
        logger.warning("No results to write")
    
    # Write errors
    if errors:
        write_errors(errors, config)
    
    logger.info(f"Complete: {len(all_results)} results, {len(errors)} errors")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python run_from_config.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    try:
        main(config_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
