"""
Tests for null models and significance testing.
"""

import numpy as np
import pytest
from ts2net.stats.null_models import (
    generate_surrogate,
    compute_network_metric_significance,
    compute_multiple_metrics_significance,
    compute_zscore,
    NetworkSignificanceResult,
)
from ts2net import HVG, NVG


class TestSurrogateGeneration:
    """Test surrogate data generation methods."""
    
    def test_shuffle_surrogate(self):
        """Test shuffle surrogate preserves values but not order."""
        x = np.array([1, 2, 3, 4, 5])
        x_surr = generate_surrogate(x, method="shuffle", rng=np.random.default_rng(42))
        
        assert len(x_surr) == len(x)
        assert set(x_surr) == set(x)  # Same values
        # With high probability, order should be different (but not guaranteed)
        assert isinstance(x_surr, np.ndarray)
    
    def test_phase_surrogate(self):
        """Test phase randomization surrogate."""
        x = np.random.randn(100)
        x_surr = generate_surrogate(x, method="phase", rng=np.random.default_rng(42))
        
        assert len(x_surr) == len(x)
        assert isinstance(x_surr, np.ndarray)
        # Phase randomization should preserve power spectrum (approximately)
        # Mean should be similar
        assert np.abs(np.mean(x_surr) - np.mean(x)) < 1.0
    
    def test_circular_surrogate(self):
        """Test circular shift surrogate."""
        x = np.array([1, 2, 3, 4, 5])
        x_surr = generate_surrogate(x, method="circular", rng=np.random.default_rng(42))
        
        assert len(x_surr) == len(x)
        assert set(x_surr) == set(x)  # Same values, just shifted
        assert isinstance(x_surr, np.ndarray)
    
    def test_iaaft_surrogate(self):
        """Test IAAFT surrogate."""
        x = np.random.randn(100)
        x_surr = generate_surrogate(x, method="iaaft", iters=10, rng=np.random.default_rng(42))
        
        assert len(x_surr) == len(x)
        assert isinstance(x_surr, np.ndarray)
    
    def test_block_bootstrap_surrogate(self):
        """Test block bootstrap surrogate."""
        x = np.random.randn(100)
        x_surr = generate_surrogate(x, method="block_bootstrap", block_size=10, rng=np.random.default_rng(42))
        
        assert len(x_surr) == len(x)
        assert isinstance(x_surr, np.ndarray)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        x = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown surrogate method"):
            generate_surrogate(x, method="invalid_method")
    
    def test_surrogate_preserves_length(self):
        """Test that all surrogate methods preserve series length."""
        x = np.random.randn(50)
        methods = ["shuffle", "phase", "circular", "iaaft", "block_bootstrap"]
        
        for method in methods:
            x_surr = generate_surrogate(x, method=method, rng=np.random.default_rng(42))
            assert len(x_surr) == len(x), f"{method} failed length preservation"


class TestZScoreComputation:
    """Test z-score computation."""
    
    def test_zscore_basic(self):
        """Test basic z-score computation."""
        observed = 5.0
        null_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        z_score, null_mean, null_std = compute_zscore(observed, null_values)
        
        assert null_mean == 3.0
        assert null_std > 0
        assert isinstance(z_score, float)
        # Observed equals max of null, so z-score should be positive
        assert z_score > 0
    
    def test_zscore_zero_std(self):
        """Test z-score with zero standard deviation."""
        observed = 5.0
        null_values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        z_score, null_mean, null_std = compute_zscore(observed, null_values)
        
        assert null_mean == 5.0
        assert null_std >= 1e-12  # Should be clamped to avoid division by zero
        assert isinstance(z_score, float)


class TestNetworkMetricSignificance:
    """Test network metric significance testing."""
    
    def test_test_network_metric_basic(self):
        """Test basic network metric significance testing."""
        x = np.random.randn(100)
        
        def compute_density(ts):
            hvg = HVG()
            hvg.build(ts)
            stats = hvg.stats()
            return stats['density']
        
        result = compute_network_metric_significance(
            x,
            compute_density,
            method="shuffle",
            n_surrogates=50,  # Use fewer for speed
            metric_name="density"
        )
        
        assert isinstance(result, NetworkSignificanceResult)
        assert result.metric_name == "density"
        assert result.n_surrogates == 50
        assert result.surrogate_method == "shuffle"
        assert isinstance(result.observed_value, float)
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)
        assert 0 <= result.p_value <= 1
        # Check that significant is a boolean
        assert result.significant in [True, False]
        assert len(result.confidence_interval) == 2
    
    def test_test_network_metric_phase(self):
        """Test significance testing with phase randomization."""
        x = np.random.randn(100)
        
        def compute_avg_degree(ts):
            hvg = HVG()
            hvg.build(ts)
            stats = hvg.stats()
            return stats['avg_degree']
        
        result = compute_network_metric_significance(
            x,
            compute_avg_degree,
            method="phase",
            n_surrogates=50,
            metric_name="avg_degree"
        )
        
        assert result.surrogate_method == "phase"
        assert isinstance(result.z_score, float)
    
    def test_test_multiple_metrics(self):
        """Test testing multiple metrics simultaneously."""
        x = np.random.randn(100)
        
        def compute_density(ts):
            hvg = HVG()
            hvg.build(ts)
            return hvg.stats()['density']
        
        def compute_std_degree(ts):
            hvg = HVG()
            hvg.build(ts)
            return hvg.stats()['std_degree']
        
        metrics = {
            "density": compute_density,
            "std_degree": compute_std_degree,
        }
        
        results = compute_multiple_metrics_significance(
            x,
            metrics,
            method="shuffle",
            n_surrogates=50
        )
        
        assert len(results) == 2
        assert "density" in results
        assert "std_degree" in results
        assert isinstance(results["density"], NetworkSignificanceResult)
        assert isinstance(results["std_degree"], NetworkSignificanceResult)


class TestNetworkClassMethods:
    """Test significance testing methods on network classes."""
    
    def test_hvg_test_significance(self):
        """Test HVG.test_significance() method."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        result = hvg.test_significance(
            metric="density",
            method="shuffle",
            n_surrogates=50
        )
        
        assert isinstance(result, NetworkSignificanceResult)
        assert result.metric_name == "density"
        assert result.n_surrogates == 50
    
    def test_hvg_test_significance_multiple_metrics(self):
        """Test HVG significance testing with different metrics."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        metrics = ["density", "avg_degree", "std_degree"]
        for metric in metrics:
            result = hvg.test_significance(
                metric=metric,
                method="shuffle",
                n_surrogates=30  # Fewer for speed
            )
            assert result.metric_name == metric
            assert isinstance(result.z_score, float)
    
    def test_hvg_test_significance_invalid_metric(self):
        """Test that invalid metric raises error."""
        x = np.random.randn(100)
        hvg = HVG()
        hvg.build(x)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            hvg.test_significance(metric="invalid_metric", n_surrogates=10)
    
    def test_hvg_test_significance_not_built(self):
        """Test that test_significance requires graph to be built."""
        hvg = HVG()
        
        with pytest.raises(ValueError):
            hvg.test_significance(metric="density", n_surrogates=10)
    
    def test_nvg_test_significance(self):
        """Test NVG.test_significance() method."""
        x = np.random.randn(100)
        nvg = NVG()
        nvg.build(x)
        
        result = nvg.test_significance(
            metric="density",
            method="phase",
            n_surrogates=50
        )
        
        assert isinstance(result, NetworkSignificanceResult)
        assert result.metric_name == "density"
        assert result.surrogate_method == "phase"
    
    def test_nvg_test_significance_block_bootstrap(self):
        """Test NVG significance testing with block bootstrap."""
        x = np.random.randn(100)
        nvg = NVG()
        nvg.build(x)
        
        result = nvg.test_significance(
            metric="density",
            method="block_bootstrap",
            n_surrogates=50,
            block_size=10
        )
        
        assert result.surrogate_method == "block_bootstrap"


class TestNetworkSignificanceResult:
    """Test NetworkSignificanceResult dataclass."""
    
    def test_result_str(self):
        """Test string representation of result."""
        result = NetworkSignificanceResult(
            metric_name="density",
            observed_value=0.5,
            null_mean=0.3,
            null_std=0.1,
            z_score=2.0,
            p_value=0.05,
            n_surrogates=200,
            surrogate_method="shuffle",
            significant=True,
            confidence_interval=(0.1, 0.5),
        )
        
        s = str(result)
        assert "density" in s
        assert "z=" in s
        assert "p=" in s
    
    def test_result_summary(self):
        """Test summary dictionary."""
        result = NetworkSignificanceResult(
            metric_name="density",
            observed_value=0.5,
            null_mean=0.3,
            null_std=0.1,
            z_score=2.0,
            p_value=0.05,
            n_surrogates=200,
            surrogate_method="shuffle",
            significant=True,
            confidence_interval=(0.1, 0.5),
        )
        
        summary = result.summary()
        assert isinstance(summary, dict)
        assert summary["metric"] == "density"
        assert summary["observed"] == 0.5
        assert summary["z_score"] == 2.0
        assert summary["significant"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_short_series(self):
        """Test with very short series."""
        x = np.random.randn(10)
        hvg = HVG()
        hvg.build(x)
        
        # Should not crash, but may have limited statistical power
        result = hvg.test_significance(
            metric="density",
            n_surrogates=20
        )
        assert isinstance(result, NetworkSignificanceResult)
    
    def test_constant_series(self):
        """Test with constant series."""
        x = np.ones(100)
        hvg = HVG()
        hvg.build(x)
        
        # Should handle constant series gracefully
        result = hvg.test_significance(
            metric="density",
            n_surrogates=20
        )
        assert isinstance(result, NetworkSignificanceResult)
    
    def test_directed_graph_significance(self):
        """Test significance testing with directed graphs."""
        x = np.random.randn(100)
        hvg = HVG(directed=True)
        hvg.build(x)
        
        result = hvg.test_significance(
            metric="density",
            n_surrogates=30
        )
        assert isinstance(result, NetworkSignificanceResult)
        
        # Test irreversibility_score if available
        stats = hvg.stats()
        if "irreversibility_score" in stats:
            result = hvg.test_significance(
                metric="irreversibility_score",
                n_surrogates=30
            )
            assert isinstance(result, NetworkSignificanceResult)

