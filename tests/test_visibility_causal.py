"""
Tests for directed visibility causal asymmetry analysis.
"""

import numpy as np
import networkx as nx
import pytest

from ts2net.causal.visibility import (
    directed_visibility_analysis,
    visibility_irreversibility,
    visibility_asymmetry_panel,
)


class TestDirectedVisibilityAnalysis:
    def test_returns_graph_and_scores(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
        result = directed_visibility_analysis(x)

        assert isinstance(result.graph, nx.DiGraph)
        assert 0.0 <= result.irreversibility_score <= 1.0
        assert len(result.in_degrees) == len(x)
        assert len(result.out_degrees) == len(x)
        assert "n_edges" in result.stats

    def test_asymmetric_pattern_detected(self):
        asymmetric = np.concatenate([np.linspace(0, 10, 120), np.zeros(120)])
        result = directed_visibility_analysis(asymmetric, compare_reversed=False)
        assert result.irreversibility_score > 0.0

    def test_temporal_asymmetry_index_nonzero_for_asymmetric_signal(self):
        x = np.concatenate([np.linspace(0, 10, 100), np.zeros(100)])
        result = directed_visibility_analysis(x, compare_reversed=True)
        assert result.temporal_asymmetry_index != 0.0

    def test_visibility_irreversibility_shorthand(self):
        x = np.linspace(0, 1, 100)
        score = visibility_irreversibility(x)
        full = directed_visibility_analysis(x, compare_reversed=False)
        assert score == pytest.approx(full.irreversibility_score)

    def test_summary_string(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 120))
        text = directed_visibility_analysis(x).summary()
        assert "Irreversibility score" in text

    def test_panel_analysis(self):
        rng = np.random.default_rng(1)
        asymmetric = np.concatenate([np.linspace(0, 5, 60), np.zeros(60)])
        noise = rng.standard_normal(120)
        X = np.vstack([asymmetric, noise])
        results = visibility_asymmetry_panel(X, axis=0)
        assert len(results) == 2
        assert all(0.0 <= r.irreversibility_score <= 1.0 for r in results.values())

    def test_rejects_invalid_input(self):
        with pytest.raises(Exception):
            directed_visibility_analysis(np.ones((3, 3)))
