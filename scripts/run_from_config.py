#!/usr/bin/env python3
"""
YAML-based pipeline runner for ts2net.

Loads configuration from YAML, validates it, and runs the analysis pipeline.
Keeps "what" in YAML, "how" in Python.
"""

import sys
import os
import yaml
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
    from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
except ImportError as e:
    logging.error(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['dataset', 'graphs', 'output']
    for section in required:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    return config


def load_series(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load time series from dataset configuration."""
    dataset = config['dataset']
    path = dataset['path']
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Use Polars if available, otherwise fallback to pandas
    if HAS_POLARS:
        logger.info(f"Loading series from {path} using Polars...")
        series = load_series_from_parquet_polars(
            path=path,
            time_col=dataset['time_col'],
            value_col=dataset['value_col'],
            id_col=dataset.get('id_col'),
            start=dataset.get('start'),
            end=dataset.get('end'),
            freq=config.get('sampling', {}).get('frequency'),
            agg=config.get('sampling', {}).get('agg', 'mean'),
            tz=dataset.get('tz')
        )
    else:
        # Fallback to pandas (would need to implement)
        raise ImportError("Polars required for data loading. Install with: pip install ts2net[polars]")
    
    logger.info(f"Loaded {len(series)} series")
    return series


def build_graph(series: np.ndarray, graph_type: str, graph_config: Dict[str, Any], 
                output_mode: str = "stats") -> Dict[str, Any]:
    """Build a single graph and return statistics."""
    # Safety check: refuse dense adjacency unless explicitly forced
    if graph_config.get('force_dense', False) and len(series) > 50_000:
        raise ValueError(
            f"Refusing dense adjacency for n={len(series)}. "
            f"This would require ~{len(series)**2 * 8 / 1e9:.1f} GB. "
            f"Use sparse matrices or output='stats' instead."
        )
    
    # Determine output mode (only_degrees is deprecated, use output instead)
    output_mode = graph_config.get('output', output_mode)
    
    try:
        if graph_type == 'hvg':
            builder = HVG(
                weighted=graph_config.get('weighted', False),
                limit=graph_config.get('limit'),
                output=output_mode
            )
            g = builder.build(series)
            
        elif graph_type == 'nvg':
            builder = NVG(
                weighted=graph_config.get('weighted', False),
                limit=graph_config.get('limit'),
                max_edges=graph_config.get('max_edges'),
                max_edges_per_node=graph_config.get('max_edges_per_node'),
                max_memory_mb=graph_config.get('max_memory_mb'),
                output=output_mode
            )
            g = builder.build(series)
            
        elif graph_type == 'recurrence':
            # Safety check: refuse exact all-pairs for large n
            n = len(series)
            rule = graph_config.get('rule', 'knn')
            if rule == 'epsilon' and n > 50_000:
                raise ValueError(
                    f"Refusing exact all-pairs recurrence for n={n}. "
                    f"Use rule='knn' with small k instead."
                )
            
            builder = RecurrenceNetwork(
                m=graph_config.get('m', 3),
                tau=graph_config.get('tau', 1),
                rule=rule,
                k=graph_config.get('k', 10),
                epsilon=graph_config.get('epsilon', 0.1),
                metric=graph_config.get('metric', 'euclidean'),
                output=output_mode
            )
            g = builder.build(series)
            
        elif graph_type == 'transition':
            builder = TransitionNetwork(
                symbolizer=graph_config.get('symbolizer', 'ordinal'),
                order=graph_config.get('order', 3),
                n_states=graph_config.get('n_states'),
                output=output_mode
            )
            g = builder.build(series)
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Get statistics
        stats = g.stats(include_triangles=graph_config.get('include_triangles', False))
        return stats
        
    except Exception as e:
        logger.warning(f"Error building {graph_type}: {e}")
        raise


def process_series(series_id: str, series_data: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single series with all enabled graph types."""
    result = {'series_id': series_id, 'n_points': len(series_data)}
    
    graphs_config = config['graphs']
    
    for graph_type in ['hvg', 'nvg', 'recurrence', 'transition']:
        graph_config = graphs_config.get(graph_type, {})
        if not graph_config.get('enabled', False):
            continue
        
        try:
            stats = build_graph(series_data, graph_type, graph_config, 
                              output_mode=config.get('output', {}).get('mode', 'stats'))
            result[graph_type] = stats
        except Exception as e:
            logger.warning(f"{series_id} {graph_type}: {e}")
            if config.get('logging', {}).get('log_errors', True):
                result[f'{graph_type}_error'] = str(e)
    
    return result


def process_windowed(series_id: str, series_data: np.ndarray, config: Dict[str, Any]) -> list:
    """Process series with sliding windows."""
    windows_config = config.get('windows', {})
    window_size = windows_config.get('size')
    step = windows_config.get('step', 1)
    
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


def write_output(results: list, config: Dict[str, Any]):
    """Write results to output file."""
    output_config = config['output']
    output_path = output_config['path']
    output_format = output_config.get('format', 'parquet')
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'parquet':
        if HAS_POLARS:
            df = pl.DataFrame(results)
            df.write_parquet(output_path)
            logger.info(f"Results written to {output_path}: {len(results)} rows")
        else:
            raise ImportError("Polars required for Parquet output")
    elif output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results written to {output_path}: {len(results)} rows")
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def write_errors(errors: list, config: Dict[str, Any]):
    """Write error log if configured."""
    logging_config = config.get('logging', {})
    if logging_config.get('log_errors', False):
        error_path = logging_config.get('error_path')
        if error_path:
            with open(error_path, 'w') as f:
                json.dump(errors, f, indent=2)
            logger.info(f"Error log written to {error_path}")


def main(config_path: str):
    """Main pipeline execution."""
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    dataset_name = config['dataset'].get('name', 'unknown')
    logger.info(f"Dataset: {dataset_name}")
    
    # Load series
    series_dict = load_series(config)
    
    # Process each series
    all_results = []
    errors = []
    
    windows_enabled = config.get('windows', {}).get('enabled', False)
    
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
