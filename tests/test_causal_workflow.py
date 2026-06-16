"""
Tests for the causal analysis workflow (0.5 milestone).
"""

import numpy as np
import pytest
import networkx as nx

from ts2net.causal import (
    search_granger_lag,
    search_te_lag,
    te_permutation_test,
    partial_granger_causality,
    conditional_te_network,
    run_causal_analysis,
    CausalWorkflowSpec,
)


statsmodels = pytest.importorskip("statsmodels")


class TestLagSearch:
    def test_granger_lag_search_finds_causal_lag(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        y = np.concatenate([[0], 0.7 * x[:-1] + 0.05 * rng.standard_normal(499)])

        best_lag, scores = search_granger_lag(x, y, max_lag=5, criterion="pvalue")
        assert best_lag in scores
        assert scores[best_lag]["p_value"] < 0.05

    def test_te_lag_search_returns_scores(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300)
        y = np.concatenate([[0], 0.5 * x[:-1] + 0.1 * rng.standard_normal(299)])

        best_lag, scores = search_te_lag(x, y, lags=[1, 2, 3])
        assert best_lag in scores
        assert all(v >= 0 for v in scores.values())


class TestConfidence:
    def test_te_permutation_independent_high_p(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(400)
        y = rng.standard_normal(400)

        result = te_permutation_test(x, y, lag=1, n_permutations=29, random_state=0)
        assert "p_value" in result
        assert result["p_value"] > 0.05

    def test_te_permutation_causal_low_p(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(600)
        y = np.concatenate([[0], 0.8 * x[:-1] + 0.02 * rng.standard_normal(599)])

        result = te_permutation_test(
            x, y, lag=1, n_permutations=49, bins=8, random_state=0
        )
        assert result["te"] >= 0
        assert result["p_value"] < 0.1


class TestConfounders:
    def test_partial_granger_with_control(self):
        rng = np.random.default_rng(3)
        z = rng.standard_normal(500)
        x = rng.standard_normal(500)
        y = np.concatenate([[0], 0.5 * x[:-1] + 0.3 * z[:-1] + 0.05 * rng.standard_normal(499)])

        raw = partial_granger_causality(x, y, [], max_lag=3)
        adjusted = partial_granger_causality(x, y, [z], max_lag=3)
        assert "p_value" in adjusted
        assert adjusted["n_controls"] == 1

    def test_conditional_te_network_shape(self):
        rng = np.random.default_rng(4)
        X = [rng.standard_normal(250) for _ in range(3)]

        G, matrix, stats = conditional_te_network(X, lag=1, bins=6)
        assert isinstance(G, nx.DiGraph)
        assert matrix.shape == (3, 3)
        assert "mean_cte" in stats


class TestCausalWorkflow:
    def test_granger_workflow_detects_edge(self):
        rng = np.random.default_rng(5)
        x1 = rng.standard_normal(450)
        x2 = np.concatenate([[0], 0.65 * x1[:-1] + 0.05 * rng.standard_normal(449)])
        x3 = rng.standard_normal(450)

        result = run_causal_analysis(
            [x1, x2, x3],
            method="granger",
            lag_search=True,
            max_lag=4,
            alpha=0.05,
            series_names=["driver", "response", "noise"],
        )

        assert isinstance(result.graph, nx.DiGraph)
        assert result.method == "granger"
        assert any(e.source == 0 and e.target == 1 and e.significant for e in result.edges)
        report = result.summary()
        assert "driver" in report
        assert "response" in report

    def test_te_workflow_with_lag_search(self):
        rng = np.random.default_rng(6)
        x1 = rng.standard_normal(400)
        x2 = np.concatenate([[0], 0.7 * x1[:-1] + 0.05 * rng.standard_normal(399)])

        spec = CausalWorkflowSpec(
            method="transfer_entropy",
            lag_search=True,
            te_lags=[1, 2],
            n_permutations=19,
            bins=6,
        )
        result = run_causal_analysis([x1, x2], spec=spec)
        assert result.method == "transfer_entropy"
        assert len(result.lag_by_pair) >= 1
        assert result.summary()

    def test_workflow_spec_kwargs(self):
        rng = np.random.default_rng(7)
        X = [rng.standard_normal(200) for _ in range(2)]
        result = run_causal_analysis(X, method="granger", lag_search=False, max_lag=2)
        assert result.metrics["lag_search"] is False
