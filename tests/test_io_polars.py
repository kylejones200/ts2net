"""
Tests for Polars-based Parquet ingestion.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    from ts2net.io_polars import load_series_from_parquet_polars
    HAS_IO_POLARS = True
except ImportError:
    HAS_IO_POLARS = False


@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample Parquet file for testing."""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    data = {
        'timestamp': dates,
        'value': np.random.randn(100).astype(np.float64),
        'meter_id': ['meter_1'] * 50 + ['meter_2'] * 50
    }
    df = pd.DataFrame(data)
    
    parquet_path = tmp_path / 'test_series.parquet'
    df.to_parquet(parquet_path, index=False)
    
    return str(parquet_path), df


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.skipif(not HAS_IO_POLARS, reason="io_polars not available")
class TestLoadSeriesFromParquetPolars:
    """Tests for load_series_from_parquet_polars."""
    
    def test_single_series(self, sample_parquet):
        """Test loading single series (no id_col)."""
        parquet_path, df_expected = sample_parquet
        
        times, values = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value'
        )
        
        assert isinstance(times, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(times) == len(values)
        assert values.dtype == np.float64
        assert len(values) == 100
    
    def test_multiple_series(self, sample_parquet):
        """Test loading multiple series with id_col."""
        parquet_path, df_expected = sample_parquet
        
        series = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value',
            id_col='meter_id'
        )
        
        assert isinstance(series, dict)
        assert len(series) == 2
        assert 'meter_1' in series
        assert 'meter_2' in series
        assert len(series['meter_1']) == 50
        assert len(series['meter_2']) == 50
        assert all(isinstance(v, np.ndarray) for v in series.values())
        assert all(v.dtype == np.float64 for v in series.values())
    
    def test_time_filtering(self, sample_parquet):
        """Test start/end time filtering."""
        parquet_path, df_expected = sample_parquet
        
        # Filter to first 25 hours
        times, values = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value',
            start='2024-01-01 00:00:00',
            end='2024-01-02 00:00:00'
        )
        
        assert len(values) == 25  # 24 hours + 1 (inclusive)
    
    def test_freq_aggregation(self, sample_parquet):
        """Test time-based frequency aggregation."""
        parquet_path, df_expected = sample_parquet
        
        # Aggregate to daily
        times, values = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value',
            freq='1d',
            agg='mean'
        )
        
        # Should have ~4-5 days of data (100 hours / 24)
        assert len(values) >= 4
        assert len(values) <= 5
    
    def test_freq_with_id_col(self, sample_parquet):
        """Test frequency aggregation with multiple series."""
        parquet_path, df_expected = sample_parquet
        
        series = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value',
            id_col='meter_id',
            freq='1d',
            agg='mean'
        )
        
        assert len(series) == 2
        # Each series should have ~2-3 days (50 hours / 24)
        assert all(2 <= len(v) <= 3 for v in series.values())
    
    def test_drop_nulls(self, tmp_path):
        """Test that nulls are dropped."""
        # Create data with nulls
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        data = {
            'timestamp': dates,
            'value': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
        }
        df = pd.DataFrame(data)
        parquet_path = tmp_path / 'test_with_nulls.parquet'
        df.to_parquet(parquet_path, index=False)
        
        times, values = load_series_from_parquet_polars(
            path=str(parquet_path),
            time_col='timestamp',
            value_col='value'
        )
        
        # Should have 8 non-null values
        assert len(values) == 8
        assert not np.any(np.isnan(values))


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.skipif(not HAS_IO_POLARS, reason="io_polars not available")
class TestPolarsVsPandas:
    """Compare Polars loader with pandas-based equivalent."""
    
    def test_polars_vs_pandas_single_series(self, sample_parquet):
        """Compare Polars and pandas results for single series."""
        parquet_path, df_expected = sample_parquet
        
        # Polars result
        times_polars, values_polars = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value'
        )
        
        # Pandas equivalent
        df_pandas = pd.read_parquet(parquet_path)
        df_pandas = df_pandas.sort_values('timestamp')
        df_pandas = df_pandas.dropna()
        values_pandas = df_pandas['value'].values.astype(np.float64)
        
        # Should match (allowing for small numerical differences)
        assert len(values_polars) == len(values_pandas)
        assert np.allclose(values_polars, values_pandas, rtol=1e-10)
    
    def test_polars_vs_pandas_multiple_series(self, sample_parquet):
        """Compare Polars and pandas results for multiple series."""
        parquet_path, df_expected = sample_parquet
        
        # Polars result
        series_polars = load_series_from_parquet_polars(
            path=parquet_path,
            time_col='timestamp',
            value_col='value',
            id_col='meter_id'
        )
        
        # Pandas equivalent
        df_pandas = pd.read_parquet(parquet_path)
        df_pandas = df_pandas.sort_values(['meter_id', 'timestamp'])
        df_pandas = df_pandas.dropna()
        series_pandas = {}
        for meter_id in df_pandas['meter_id'].unique():
            meter_data = df_pandas[df_pandas['meter_id'] == meter_id]
            series_pandas[meter_id] = meter_data['value'].values.astype(np.float64)
        
        # Should match
        assert set(series_polars.keys()) == set(series_pandas.keys())
        for meter_id in series_polars.keys():
            assert len(series_polars[meter_id]) == len(series_pandas[meter_id])
            assert np.allclose(
                series_polars[meter_id],
                series_pandas[meter_id],
                rtol=1e-10
            )

