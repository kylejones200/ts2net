"""
Polars-based Parquet ingestion for time series data.

This module provides efficient lazy-loading of time series from Parquet files
using Polars, converting to NumPy arrays for use with ts2net core algorithms.
"""

from __future__ import annotations

from typing import Optional, Union, Dict, Tuple
import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


def load_series_from_parquet_polars(
    path: str,
    time_col: str,
    value_col: str,
    id_col: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: Optional[str] = None,
    agg: str = "mean",
    tz: Optional[str] = None,
    columns_extra: Optional[list[str]] = None,
) -> Union[Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load time series from Parquet file using Polars (lazy evaluation).
    
    Uses lazy evaluation to minimize memory usage. Converts to NumPy arrays
    for compatibility with ts2net core algorithms.
    
    Parameters
    ----------
    path : str
        Path to Parquet file or directory of Parquet files
    time_col : str
        Column name for timestamps
    value_col : str
        Column name for values
    id_col : str, optional
        Column name for series identifier (e.g., meter_id, region)
        If None, returns single series as tuple (times, values)
    start : str, optional
        Start timestamp filter (ISO format or parseable by Polars)
    end : str, optional
        End timestamp filter (ISO format or parseable by Polars)
    freq : str, optional
        Time frequency for bucketing (e.g., '1h', '1d', '15m')
        Uses Polars group_by_dynamic for efficient time-based aggregation
    agg : str, default 'mean'
        Aggregation function: 'mean', 'sum', 'min', 'max', 'median', 'first', 'last'
    tz : str, optional
        Timezone for time_col (e.g., 'UTC', 'Europe/Madrid')
    columns_extra : list[str], optional
        Additional columns to keep in output (not used for aggregation)
    
    Returns
    -------
    dict[str, np.ndarray] or tuple[np.ndarray, np.ndarray]
        If id_col is provided: dict mapping id -> values array
        If id_col is None: tuple of (times, values) arrays
        
    Examples
    --------
    >>> # Single series
    >>> times, values = load_series_from_parquet_polars(
    ...     'data.parquet', time_col='timestamp', value_col='consumption'
    ... )
    
    >>> # Multiple series by meter_id
    >>> series = load_series_from_parquet_polars(
    ...     'data.parquet',
    ...     time_col='timestamp',
    ...     value_col='consumption',
    ...     id_col='meter_id',
    ...     freq='1h',
    ...     start='2024-01-01',
    ...     end='2024-12-31'
    ... )
    >>> # series = {'meter_1': np.array([...]), 'meter_2': np.array([...]), ...}
    """
    if not HAS_POLARS:
        raise ImportError(
            "Polars is required for load_series_from_parquet_polars. "
            "Install with: pip install ts2net[polars]"
        )
    
    # Build column selection
    select_cols = [time_col, value_col]
    if id_col is not None:
        select_cols.append(id_col)
    if columns_extra:
        select_cols.extend(columns_extra)
    
    # Lazy scan - only reads metadata initially
    df = pl.scan_parquet(path).select(select_cols)
    
    # Apply timezone if specified (convert string time_col to datetime if needed)
    # Note: Polars will handle timezone conversion if time_col is already datetime
    if tz is not None:
        # Try to set timezone - will work if column is datetime
        try:
            df = df.with_columns(
                pl.col(time_col).dt.replace_time_zone(tz)
            )
        except Exception:
            # If time_col is string, parse it first
            df = df.with_columns(
                pl.col(time_col).str.strptime(pl.Datetime).dt.replace_time_zone(tz)
            )
    
    # Apply time filters with pushdown (efficient)
    # Convert string timestamps to Python datetime objects for comparison
    if start is not None:
        if isinstance(start, str):
            from datetime import datetime
            try:
                # Try ISO format first (handles '2024-01-01 00:00:00' and '2024-01-01T00:00:00')
                start_parsed = datetime.fromisoformat(start.replace('Z', '+00:00'))
            except ValueError:
                # Try common format 'YYYY-MM-DD HH:MM:SS'
                try:
                    start_parsed = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Try date only
                    start_parsed = datetime.strptime(start, "%Y-%m-%d")
            df = df.filter(pl.col(time_col) >= start_parsed)
        else:
            df = df.filter(pl.col(time_col) >= start)
    if end is not None:
        if isinstance(end, str):
            from datetime import datetime
            try:
                # Try ISO format first
                end_parsed = datetime.fromisoformat(end.replace('Z', '+00:00'))
            except ValueError:
                # Try common format 'YYYY-MM-DD HH:MM:SS'
                try:
                    end_parsed = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Try date only
                    end_parsed = datetime.strptime(end, "%Y-%m-%d")
            df = df.filter(pl.col(time_col) <= end_parsed)
        else:
            df = df.filter(pl.col(time_col) <= end)
    
    # Map aggregation function name to Polars method
    agg_map = {
        'mean': lambda col: col.mean(),
        'sum': lambda col: col.sum(),
        'min': lambda col: col.min(),
        'max': lambda col: col.max(),
        'median': lambda col: col.median(),
        'first': lambda col: col.first(),
        'last': lambda col: col.last(),
    }
    agg_func = agg_map.get(agg.lower(), agg_map['mean'])
    
    # Time-based bucketing if freq is specified
    if freq is not None:
        if id_col is not None:
            # Group by id_col and time buckets using group_by_dynamic with by parameter
            df = (
                df
                .sort([id_col, time_col])  # Must be sorted for group_by_dynamic
                .group_by_dynamic(
                    time_col,
                    every=freq,
                    closed="left",
                    label="left",
                    by=id_col
                )
                .agg(
                    agg_func(pl.col(value_col)).alias(value_col)
                )
            )
        else:
            # Single series with time bucketing
            df = (
                df
                .sort(time_col)  # Must be sorted for group_by_dynamic
                .group_by_dynamic(
                    time_col,
                    every=freq,
                    closed="left",
                    label="left"
                )
                .agg(
                    agg_func(pl.col(value_col)).alias(value_col)
                )
            )
    elif id_col is not None:
        # Group by id_col and time_col (no time bucketing, just deduplication)
        df = (
            df
            .group_by([id_col, time_col])
            .agg(
                agg_func(pl.col(value_col)).alias(value_col)
            )
        )
    
    # Sort by time (and id_col if present)
    sort_cols = [id_col, time_col] if id_col is not None else [time_col]
    df = df.sort(sort_cols)
    
    # Materialize and convert to NumPy
    # Only materialize what we need - lazy until here
    df_materialized = df.collect()
    
    # Drop nulls
    df_materialized = df_materialized.drop_nulls()
    
    if id_col is None:
        # Single series - return (times, values)
        # Note: times may be datetime, convert to numpy array
        times_arr = df_materialized[time_col].to_numpy()
        values = df_materialized[value_col].to_numpy().astype(np.float64)
        return times_arr, values
    else:
        # Multiple series - group by id_col
        result = {}
        unique_ids = df_materialized[id_col].unique().to_list()
        for id_val in unique_ids:
            series_df = df_materialized.filter(pl.col(id_col) == id_val)
            # Already sorted by time from earlier sort
            values = series_df[value_col].to_numpy().astype(np.float64)
            result[str(id_val)] = values
        return result
