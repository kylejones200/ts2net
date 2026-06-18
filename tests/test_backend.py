"""Tests for unified compute backend and visibility degree stats."""

from __future__ import annotations

import numpy as np
import pytest

from ts2net.config import HVGConfig, NVGConfig, RecurrenceConfig
from ts2net.core.backend import resolve_compute_backend, rust_available
from ts2net.core.recurrence_backend import recurrence_degree_stats
from ts2net.core.visibility_backend import visibility_degree_stats
from ts2net.factory import create_graph_builder


class TestBackendSelector:
    def test_resolve_auto_prefers_rust_when_available(self):
        backend = resolve_compute_backend("auto")
        if rust_available():
            assert backend == "rust"
        else:
            assert backend in ("numba", "python")

    def test_resolve_explicit_python(self):
        assert resolve_compute_backend("python") == "python"


class TestVisibilityDegreeStats:
    def test_hvg_matches_builder_stats(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        config = HVGConfig(enabled=True, output="stats")
        fast = visibility_degree_stats(x, "hvg", backend="auto")
        assert fast is not None
        builder = create_graph_builder("hvg", config, n_points=len(x))
        builder.build(x)
        full = builder.stats()
        assert fast["n_nodes"] == full["n_nodes"]
        assert fast["n_edges"] == full["n_edges"]
        assert fast["avg_degree"] == pytest.approx(full["avg_degree"])

    def test_nvg_with_limit(self):
        x = np.sin(np.linspace(0, 10 * np.pi, 300))
        config = NVGConfig(enabled=True, output="stats", limit=50)
        fast = visibility_degree_stats(x, "nvg", limit=50, backend="auto")
        assert fast is not None
        builder = create_graph_builder("nvg", config, n_points=len(x))
        builder.build(x)
        full = builder.stats()
        assert fast["n_edges"] == full["n_edges"]


class TestRecurrenceDegreeStats:
    def test_knn_matches_builder_stats(self):
        pytest.importorskip("ts2net_rs")
        rng = np.random.default_rng(3)
        x = rng.standard_normal(120)
        config = RecurrenceConfig(enabled=True, output="stats", rule="knn", k=5)
        fast = recurrence_degree_stats(
            x, rule="knn", k=5, backend="rust"
        )
        assert fast is not None
        builder = create_graph_builder("recurrence", config, n_points=len(x))
        builder.build(x)
        full = builder.stats()
        assert fast["n_nodes"] == full["n_nodes"]
        assert fast["n_edges"] == full["n_edges"]

    def test_epsilon_matches_builder_stats(self):
        pytest.importorskip("ts2net_rs")
        x = np.sin(np.linspace(0, 6 * np.pi, 100))
        config = RecurrenceConfig(
            enabled=True, output="stats", rule="epsilon", epsilon=0.35, k=5
        )
        fast = recurrence_degree_stats(
            x, rule="epsilon", epsilon=0.35, backend="rust"
        )
        assert fast is not None
        builder = create_graph_builder("recurrence", config, n_points=len(x))
        builder.build(x)
        full = builder.stats()
        assert fast["n_nodes"] == full["n_nodes"]
        assert fast["n_edges"] == full["n_edges"]


class TestChunkedDtw:
    def test_cdist_dtw_chunked_matches_full(self):
        pytest.importorskip("ts2net_rs")
        from ts2net.distances.dtw import cdist_dtw, cdist_dtw_chunked

        rng = np.random.default_rng(1)
        X = rng.standard_normal((80, 40))
        D_full = cdist_dtw(X, backend="rust")
        D_chunk = cdist_dtw_chunked(X, chunk_size=16, backend="rust")
        np.testing.assert_allclose(D_full, D_chunk, rtol=1e-10, atol=1e-10)

    def test_ts_dist_dtw_large_panel(self):
        pytest.importorskip("ts2net_rs")
        from ts2net.multivariate.distances import ts_dist

        rng = np.random.default_rng(2)
        X = rng.standard_normal((70, 30))
        D = ts_dist(X, method="dtw", panel_chunk_threshold=32, chunk_size=16)
        assert D.shape == (70, 70)
        assert np.allclose(np.diag(D), 0.0)
