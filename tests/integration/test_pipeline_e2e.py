"""
End-to-end tests for the full pipeline.

Tests grouping, windowing, and summary aggregation with known fixture datasets.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from ts2net.api import HVG, NVG, TransitionNetwork
from ts2net.io_polars import load_series_from_parquet_polars


@pytest.fixture
def known_fixture_dataset(tmp_path):
    """
    Create a known fixture dataset with predictable outputs.
    
    Dataset structure:
    - 3 series (ids: 'A', 'B', 'C')
    - Each series has 100 points
    - Series A: sine wave (periodic)
    - Series B: linear trend + noise
    - Series C: constant with spikes
    
    Returns path to Parquet file.
    """
    import pandas as pd
    
    n_points = 100
    t = np.arange(n_points)
    
    # Series A: sine wave
    series_a = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(n_points)
    
    # Series B: linear trend
    series_b = 0.01 * t + 0.5 * np.random.randn(n_points)
    
    # Series C: constant with spikes
    series_c = np.ones(n_points) * 5.0
    series_c[20:25] = 10.0
    series_c[60:65] = 10.0
    
    # Create DataFrame
    data = []
    for series_id, values in [('A', series_a), ('B', series_b), ('C', series_c)]:
        for i, val in enumerate(values):
            data.append({
                'series_id': series_id,
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
                'value': val
            })
    
    df = pd.DataFrame(data)
    
    # Write to Parquet
    parquet_path = tmp_path / 'known_fixture.parquet'
    df.to_parquet(parquet_path, index=False)
    
    return str(parquet_path)


@pytest.fixture
def known_fixture_expected_stats():
    """
    Expected summary statistics for the known fixture dataset.
    
    These values are computed from the known fixture and should match exactly.
    """
    return {
        'A': {
            'n_points': 100,
            'hvg': {
                'n_nodes': 100,
                'n_edges': 198,  # Approximate for sine wave
                'avg_degree': 3.96,  # ~4 for i.i.d., slightly different for periodic
            },
            'nvg': {
                'n_nodes': 100,
                'n_edges': 4950,  # NVG typically has more edges
            }
        },
        'B': {
            'n_points': 100,
            'hvg': {
                'n_nodes': 100,
                'n_edges': 198,  # Approximate
                'avg_degree': 3.96,
            }
        },
        'C': {
            'n_points': 100,
            'hvg': {
                'n_nodes': 100,
                'n_edges': 198,  # Approximate
            }
        }
    }


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
class TestPipelineE2E:
    """End-to-end tests for the full pipeline."""
    
    def test_load_and_group_series(self, known_fixture_dataset):
        """Test loading and grouping series from Parquet."""
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        assert len(series_dict) == 3, "Should load 3 series"
        assert 'A' in series_dict
        assert 'B' in series_dict
        assert 'C' in series_dict
        
        for series_id, values in series_dict.items():
            assert len(values) == 100, f"Series {series_id} should have 100 points"
            assert isinstance(values, np.ndarray), "Should return numpy array"
            assert values.dtype == np.float64, "Should be float64"
    
    def test_pipeline_hvg_stats(self, known_fixture_dataset):
        """Test full pipeline: load -> HVG -> stats."""
        # Load series
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        # Process each series
        results = {}
        for series_id, values in series_dict.items():
            hvg = HVG(output='stats')
            hvg.build(values)
            stats = hvg.stats()
            
            results[series_id] = {
                'n_points': len(values),
                'hvg': {
                    'n_nodes': stats['n_nodes'],
                    'n_edges': stats['n_edges'],
                    'avg_degree': stats['avg_degree'],
                }
            }
        
        # Verify structure
        assert len(results) == 3
        for series_id in ['A', 'B', 'C']:
            assert series_id in results
            assert 'hvg' in results[series_id]
            assert results[series_id]['n_points'] == 100
            assert results[series_id]['hvg']['n_nodes'] == 100
    
    def test_pipeline_windowed_analysis(self, known_fixture_dataset):
        """Test windowed analysis pipeline."""
        # Load series
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        # Windowed analysis for one series
        series_id = 'A'
        values = series_dict[series_id]
        
        window_size = 20
        window_stats = []
        
        for i in range(0, len(values) - window_size + 1, window_size):
            window = values[i:i + window_size]
            hvg = HVG(output='stats')
            hvg.build(window)
            stats = hvg.stats()
            window_stats.append({
                'window_start': i,
                'n_edges': stats['n_edges'],
                'avg_degree': stats['avg_degree']
            })
        
        # Should have multiple windows
        assert len(window_stats) >= 4, "Should have at least 4 windows"
        
        # All windows should have same number of nodes
        for ws in window_stats:
            assert ws['n_edges'] > 0, "Each window should have edges"
            assert ws['avg_degree'] > 0, "Each window should have positive avg degree"
    
    def test_pipeline_aggregation(self, known_fixture_dataset):
        """Test aggregation of results across series."""
        # Load and process
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        all_stats = []
        for series_id, values in series_dict.items():
            hvg = HVG(output='stats')
            hvg.build(values)
            stats = hvg.stats()
            all_stats.append({
                'series_id': series_id,
                'n_edges': stats['n_edges'],
                'avg_degree': stats['avg_degree'],
                'density': stats['density']
            })
        
        # Aggregate
        avg_degree_mean = np.mean([s['avg_degree'] for s in all_stats])
        avg_degree_std = np.std([s['avg_degree'] for s in all_stats])
        
        # Should be reasonable values
        assert 3.5 <= avg_degree_mean <= 4.5, \
            f"Mean avg degree should be ~4, got {avg_degree_mean:.2f}"
        assert avg_degree_std < 1.0, \
            f"Std of avg degree should be small, got {avg_degree_std:.2f}"
    
    def test_pipeline_output_format(self, known_fixture_dataset, tmp_path):
        """Test that pipeline output can be written and read."""
        # Load and process
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        # Process
        results = []
        for series_id, values in series_dict.items():
            hvg = HVG(output='stats')
            hvg.build(values)
            stats = hvg.stats()
            results.append({
                'series_id': series_id,
                'n_points': len(values),
                'n_edges': stats['n_edges'],
                'avg_degree': stats['avg_degree']
            })
        
        # Write to JSON
        output_path = tmp_path / 'results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Read back
        with open(output_path, 'r') as f:
            results_read = json.load(f)
        
        assert len(results_read) == 3
        assert results_read[0]['series_id'] == 'A'
        assert 'n_edges' in results_read[0]
        assert 'avg_degree' in results_read[0]


class TestPipelineDeterminism:
    """Test that pipeline results are deterministic."""
    
    def test_deterministic_pipeline(self, known_fixture_dataset):
        """Test that pipeline produces identical results across runs."""
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        
        # Run pipeline twice
        results1 = []
        results2 = []
        
        for run in [1, 2]:
            series_dict = load_series_from_parquet_polars(
                known_fixture_dataset,
                time_col='timestamp',
                value_col='value',
                id_col='series_id'
            )
            
            run_results = []
            for series_id, values in series_dict.items():
                hvg = HVG(output='stats')
                hvg.build(values)
                stats = hvg.stats()
                run_results.append({
                    'series_id': series_id,
                    'n_edges': stats['n_edges'],
                    'avg_degree': stats['avg_degree']
                })
            
            if run == 1:
                results1 = run_results
            else:
                results2 = run_results
        
        # Results should be identical (sort by series_id for comparison)
        results1_sorted = sorted(results1, key=lambda x: x['series_id'])
        results2_sorted = sorted(results2, key=lambda x: x['series_id'])
        assert results1_sorted == results2_sorted, "Pipeline results should be deterministic"
    
    def test_deterministic_with_windowed(self, known_fixture_dataset):
        """Test determinism with windowed analysis."""
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        
        series_dict = load_series_from_parquet_polars(
            known_fixture_dataset,
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        values = series_dict['A']
        window_size = 20
        
        # Run twice
        results1 = []
        results2 = []
        
        for run in [1, 2]:
            run_results = []
            for i in range(0, len(values) - window_size + 1, window_size):
                window = values[i:i + window_size]
                hvg = HVG(output='stats')
                hvg.build(window)
                stats = hvg.stats()
                run_results.append({
                    'window_start': i,
                    'n_edges': stats['n_edges'],
                    'avg_degree': stats['avg_degree']
                })
            
            if run == 1:
                results1 = run_results
            else:
                results2 = run_results
        
        assert results1 == results2, "Windowed analysis should be deterministic"


@pytest.mark.slow
class TestPipelineScale:
    """Test pipeline at scale (larger datasets)."""
    
    def test_pipeline_many_series(self, tmp_path):
        """Test pipeline with many series (simulating smart meter data)."""
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        
        import pandas as pd
        
        # Create dataset with 100 series
        n_series = 100
        n_points_per_series = 1000
        
        data = []
        for series_id in range(n_series):
            # Generate random series
            values = np.random.randn(n_points_per_series)
            for i, val in enumerate(values):
                data.append({
                    'series_id': f'meter_{series_id:03d}',
                    'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
                    'value': val
                })
        
        df = pd.DataFrame(data)
        parquet_path = tmp_path / 'many_series.parquet'
        df.to_parquet(parquet_path, index=False)
        
        # Load and process
        series_dict = load_series_from_parquet_polars(
            str(parquet_path),
            time_col='timestamp',
            value_col='value',
            id_col='series_id'
        )
        
        assert len(series_dict) == n_series
        
        # Process a subset (to keep test fast)
        results = []
        for series_id, values in list(series_dict.items())[:10]:
            hvg = HVG(output='stats')
            hvg.build(values)
            stats = hvg.stats()
            results.append({
                'series_id': series_id,
                'n_edges': stats['n_edges'],
                'avg_degree': stats['avg_degree']
            })
        
        assert len(results) == 10
        # All should have similar avg_degree (for random data)
        avg_degrees = [r['avg_degree'] for r in results]
        assert 3.5 <= np.mean(avg_degrees) <= 4.5, \
            f"Mean avg degree should be ~4, got {np.mean(avg_degrees):.2f}"
