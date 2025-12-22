"""
Columnar data adapters for ts2net.

Provides thin adapters that convert pandas/polars DataFrames to NumPy arrays
for use with ts2net core algorithms. Core algorithms remain pure NumPy.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Dict
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


def from_pandas(
    df: 'pd.DataFrame',
    value_col: str,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    sort_by_time: bool = True
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert pandas DataFrame to NumPy arrays for ts2net.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    value_col : str
        Column name for time series values
    group_col : str, optional
        Column name for grouping (e.g., meter_id, region)
        If provided, returns dict mapping group -> values array
    time_col : str, optional
        Column name for timestamps (used for sorting only)
    sort_by_time : bool, default True
        If True and time_col provided, sort by time
    
    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        If group_col is None: single array of values
        If group_col is provided: dict mapping group -> values array
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=100, freq='1h'),
    ...                    'consumption': np.random.randn(100),
    ...                    'meter_id': ['meter_1'] * 100})
    >>> # Single series
    >>> values = from_pandas(df, value_col='consumption', time_col='timestamp')
    >>> # Multiple series
    >>> series = from_pandas(df, value_col='consumption', group_col='meter_id', time_col='timestamp')
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for from_pandas. "
            "Install with: pip install pandas"
        )
    
    if group_col is None:
        # Single series
        if time_col and sort_by_time:
            df = df.sort_values(time_col)
        values = df[value_col].values.astype(np.float64)
        # Drop nulls
        mask = ~np.isnan(values)
        return values[mask]
    else:
        # Multiple series grouped by group_col
        result = {}
        if time_col and sort_by_time:
            df = df.sort_values([group_col, time_col])
        
        for group_val, group_df in df.groupby(group_col):
            values = group_df[value_col].values.astype(np.float64)
            # Drop nulls
            mask = ~np.isnan(values)
            result[str(group_val)] = values[mask]
        
        return result


def from_polars(
    df: 'pl.DataFrame',
    value_col: str,
    group_col: Optional[str] = None,
    time_col: Optional[str] = None,
    sort_by_time: bool = True
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert polars DataFrame to NumPy arrays for ts2net.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    value_col : str
        Column name for time series values
    group_col : str, optional
        Column name for grouping (e.g., meter_id, region)
        If provided, returns dict mapping group -> values array
    time_col : str, optional
        Column name for timestamps (used for sorting only)
    sort_by_time : bool, default True
        If True and time_col provided, sort by time
    
    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
        If group_col is None: single array of values
        If group_col is provided: dict mapping group -> values array
    
    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     'timestamp': pl.datetime_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), '1h', eager=True),
    ...     'consumption': np.random.randn(97),
    ...     'meter_id': ['meter_1'] * 97
    ... })
    >>> # Single series
    >>> values = from_polars(df, value_col='consumption', time_col='timestamp')
    >>> # Multiple series
    >>> series = from_polars(df, value_col='consumption', group_col='meter_id', time_col='timestamp')
    """
    if not HAS_POLARS:
        raise ImportError(
            "polars is required for from_polars. "
            "Install with: pip install ts2net[polars]"
        )
    
    if group_col is None:
        # Single series
        if time_col and sort_by_time:
            df = df.sort(time_col)
        values = df[value_col].to_numpy().astype(np.float64)
        # Drop nulls
        mask = ~np.isnan(values)
        return values[mask]
    else:
        # Multiple series grouped by group_col
        result = {}
        if time_col and sort_by_time:
            df = df.sort([group_col, time_col])
        
        # Group by group_col and extract values
        groups = df.group_by(group_col, maintain_order=True)
        for group_name, group_df in groups:
            values = group_df[value_col].to_numpy().astype(np.float64)
            # Drop nulls
            mask = ~np.isnan(values)
            result[str(group_name)] = values[mask]
        
        return result
