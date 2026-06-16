"""
Tests for Granger causality, causal metrics, and time-lagged analysis.
"""

import numpy as np
import pytest
import networkx as nx

from ts2net.causal.granger import granger_causality, granger_causality_network
from ts2net.causal.metrics import (
    causal_strength,
    directionality_index,
    causal_network_metrics,
)
from ts2net.causal.time_lagged import time_lagged_causality_network


statsmodels = pytest.importorskip("statsmodels")


class TestGrangerCausality:
    def test_independent_series_high_p_value(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.random.randn(500)

        result = granger_causality(x, y, max_lag=3, method="linear")

        assert "p_value" in result
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_causal_series_low_p_value(self):
        np.random.seed(42)
        x = np.random.randn(500)
        y = np.concatenate([[0], 0.7 * x[:-1] + 0.05 * np.random.randn(499)])

        result = granger_causality(x, y, max_lag=3, method="linear")

        assert result["p_value"] < 0.05
        assert result["significant"] is True
        assert result["f_stat"] > 0

    def test_nonlinear_method_runs(self):
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.concatenate([[0], 0.7 * x[:-1] + 0.05 * np.random.randn(199)])

        result = granger_causality(
            x, y, max_lag=2, method="nonlinear", n_permutations=19, random_state=0
        )

        assert 0.0 <= result["p_value"] <= 1.0
        assert "mse_improvement" in result

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            granger_causality(np.ones(10), np.ones(11))


class TestGrangerCausalityNetwork:
    def test_network_structure(self):
        np.random.seed(42)
        x1 = np.random.randn(400)
        x2 = np.concatenate([[0], 0.6 * x1[:-1] + 0.05 * np.random.randn(399)])
        x3 = np.random.randn(400)

        G, p_matrix, stats = granger_causality_network(
            [x1, x2, x3], max_lag=3, alpha=0.05
        )

        assert isinstance(G, nx.DiGraph)
        assert p_matrix.shape == (3, 3)
        assert np.all(np.diag(p_matrix) == 1.0)
        assert "n_edges" in stats
        assert G.has_edge(0, 1)

    def test_parallel_matches_serial(self):
        np.random.seed(7)
        X = [np.random.randn(250) for _ in range(3)]

        _, p_serial, _ = granger_causality_network(X, max_lag=2, n_jobs=1)
        _, p_parallel, _ = granger_causality_network(X, max_lag=2, n_jobs=2)

        np.testing.assert_allclose(p_serial, p_parallel, rtol=1e-10)


class TestCausalMetrics:
    def _chain_graph(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=0.8)
        G.add_edge(1, 2, weight=0.5)
        return G

    def test_causal_strength_path(self):
        G = self._chain_graph()
        strength = causal_strength(G, 0, 2)
        assert strength == pytest.approx(0.4, rel=1e-6)

    def test_directionality_emitter_receiver(self):
        G = self._chain_graph()
        di = directionality_index(G)
        assert di[0] > 0
        assert di[2] < 0

    def test_causal_network_metrics(self):
        G = self._chain_graph()
        metrics = causal_network_metrics(G)
        assert metrics["n_nodes"] == 3
        assert metrics["n_edges"] == 2
        assert metrics["top_emitters"][0] == 0
        assert 2 in metrics["top_receivers"]


class TestTimeLaggedCausality:
    def test_per_lag_transfer_entropy(self):
        np.random.seed(42)
        X = [np.random.randn(300) for _ in range(3)]

        results = time_lagged_causality_network(
            X, lags=[1, 2], method="transfer_entropy", combine="per_lag"
        )

        assert set(results.keys()) == {1, 2}
        for lag, (G, matrix, stats) in results.items():
            assert isinstance(G, nx.DiGraph)
            assert matrix.shape == (3, 3)
            assert stats["lag"] == lag

    def test_combined_max(self):
        np.random.seed(42)
        X = [np.random.randn(300) for _ in range(3)]

        G, matrix, stats = time_lagged_causality_network(
            X,
            lags=[1, 2],
            method="transfer_entropy",
            combine="max",
        )

        assert isinstance(G, nx.DiGraph)
        assert matrix.shape == (3, 3)
        assert stats["combine"] == "max"

    def test_per_lag_granger(self):
        np.random.seed(42)
        x1 = np.random.randn(350)
        x2 = np.concatenate([[0], 0.5 * x1[:-1] + 0.05 * np.random.randn(349)])

        results = time_lagged_causality_network(
            [x1, x2],
            lags=[2, 3],
            method="granger",
            combine="per_lag",
            alpha=0.05,
        )

        assert 2 in results and 3 in results
