"""
Tests for v0.4 core graph expansion (ts2net.graphs).
"""

from __future__ import annotations

import numpy as np
import pytest
import networkx as nx

from ts2net.graphs import (
    correlation_matrix,
    correlation_network,
    partial_correlation_network,
    rolling_correlation_matrix,
    similarity_network,
    RollingGraphSequence,
    graph_churn,
    edge_persistence,
    multiplex_graph,
    multiplex_visibility_graph,
    adaptive_recurrence_network,
    cross_recurrence_network,
    recurrence_quantification,
    event_sequence_network,
    event_sync_network,
    sax_symbolize,
    entropy_max_symbolize,
    sax_transition_network,
)
from ts2net.graphs.distances_extra import (
    matrix_profile_distance,
    soft_dtw_distance,
)


class TestCorrelationNetworks:
    def test_correlation_matrix_symmetric(self):
        X = np.random.randn(5, 80)
        C = correlation_matrix(X, method="pearson")
        assert C.shape == (5, 5)
        np.testing.assert_allclose(C, C.T)
        np.testing.assert_allclose(np.diag(C), 1.0)

    def test_spearman_and_kendall(self):
        X = np.random.randn(4, 60)
        C_s = correlation_matrix(X, method="spearman")
        C_k = correlation_matrix(X, method="kendall")
        assert np.all(np.abs(C_s) <= 1.0)
        assert np.all(np.abs(C_k) <= 1.0)

    def test_correlation_network_knn(self):
        X = np.random.randn(6, 50)
        G, C, D = correlation_network(X, method="pearson", rule="knn", k=2)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 6
        assert C.shape == (6, 6)
        assert D.shape == (6, 6)

    def test_rolling_correlation(self):
        x = np.random.randn(200)
        y = x + 0.1 * np.random.randn(200)
        vals, centers = rolling_correlation_matrix(x, y, window=30, step=5)
        assert len(vals) == len(centers)
        assert np.mean(vals) > 0.5

    def test_partial_correlation_network(self):
        X = np.random.randn(4, 120)
        G, P, D = partial_correlation_network(X, rule="knn", k=2)
        assert G.number_of_nodes() == 4
        assert P.shape == (4, 4)
        np.testing.assert_allclose(np.diag(P), 1.0)


class TestSimilarityNetworks:
    def test_euclidean_similarity_network(self):
        X = np.random.randn(5, 40)
        G, D = similarity_network(X, method="euclidean", rule="knn", k=2)
        assert G.number_of_nodes() == 5
        assert D.shape == (5, 5)

    def test_matrix_profile_distance(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 80))
        y = np.roll(x, 5)
        z = np.random.randn(80)
        assert matrix_profile_distance(x, y, subseq_len=8) < matrix_profile_distance(
            x, z, subseq_len=8
        )

    def test_soft_dtw_and_matrix_profile_networks(self):
        X = np.random.randn(4, 50)
        G_mp, D_mp = similarity_network(
            X, method="matrix_profile", rule="knn", k=2, subseq_len=5
        )
        assert G_mp.number_of_nodes() == 4
        G_sd, D_sd = similarity_network(X, method="soft_dtw", rule="knn", k=2)
        assert G_sd.number_of_nodes() == 4
        assert soft_dtw_distance(X[0], X[0]) == pytest.approx(0.0, abs=1e-6)


class TestDynamicGraphs:
    def test_rolling_graph_sequence(self):
        x = np.random.randn(300)
        seq = RollingGraphSequence.from_series(
            x, window=40, step=20, method="hvg", output="stats"
        )
        assert len(seq.stats) > 0
        assert len(seq.graphs_nx) == len(seq.stats)
        assert seq.stat_series("n_nodes").shape[0] == len(seq.stats)

    def test_churn_and_persistence(self):
        G1 = nx.path_graph(5)
        G2 = nx.path_graph(5)
        G2.add_edge(0, 4)
        churn = graph_churn([G1, G2])
        assert churn["births"][0] == 1
        assert churn["deaths"][0] == 0
        pers = edge_persistence([G1, G2])
        assert pers[(0, 1)] == 1.0


class TestMultiplexVisibility:
    def test_multiplex_visibility_layers(self):
        x = np.random.randn(100)
        mg, layers = multiplex_visibility_graph(x)
        assert set(mg.layer_names()) == {"hvg", "nvg"}
        assert layers["hvg"].number_of_nodes() == 100
        assert layers["nvg"].number_of_nodes() == 100

    def test_multiplex_aggregate(self):
        G1 = nx.Graph([(0, 1), (1, 2)])
        G2 = nx.Graph([(1, 2), (2, 3)])
        mg = multiplex_graph({"a": G1, "b": G2})
        A_union = mg.aggregate_adjacency("union")
        A_inter = mg.aggregate_adjacency("intersection")
        assert A_union[1, 2] > 0
        assert A_inter[1, 2] > 0
        assert A_inter[0, 1] == 0


class TestRecurrenceExpansion:
    def test_adaptive_recurrence(self):
        x = np.sin(np.linspace(0, 6 * np.pi, 120)) + 0.05 * np.random.randn(120)
        builder, eps = adaptive_recurrence_network(x, target_density=0.1)
        assert eps > 0
        assert builder.n_nodes > 0

    def test_cross_recurrence(self):
        x = np.random.randn(80)
        y = x + 0.1 * np.random.randn(80)
        G, R = cross_recurrence_network(x, y, target_density=0.2)
        assert G.number_of_nodes() == 80
        assert R.shape == (80, 80)

    def test_recurrence_quantification(self):
        x = np.sin(np.linspace(0, 8 * np.pi, 150)) + 0.02 * np.random.randn(150)
        result = recurrence_quantification(x, target_density=0.15)
        assert "rqa" in result
        assert "RR" in result["rqa"]
        assert "DET" in result["rqa"]
        assert 0.0 <= result["recurrence_rate"] <= 1.0
        assert result["builder"].n_nodes > 0


class TestEventNetworks:
    def test_event_sequence_network(self):
        x = np.zeros(100)
        x[10] = 5.0
        x[25] = 4.0
        x[30] = 6.0
        G, events = event_sequence_network(
            x, method="threshold", thresh=1.0, edge_rule="window", max_interval=25
        )
        assert len(events) >= 2
        assert G.number_of_edges() >= 1

    def test_event_sync_network(self):
        np.random.seed(0)
        X = np.random.randn(4, 200)
        X[0, 50] = 10.0
        X[1, 52] = 9.0
        G, sync, event_sets = event_sync_network(
            X, method="threshold", thresh=2.0, rule="complete"
        )
        assert G.number_of_nodes() == 4
        assert sync.shape == (4, 4)
        assert len(event_sets) == 4


class TestTransitionExpansion:
    def test_sax_symbolize(self):
        x = np.random.randn(90)
        symbols = sax_symbolize(x, n_bins=5, word_size=3)
        assert len(symbols) == 30
        assert symbols.min() >= 0

    def test_entropy_max_and_sax_network(self):
        x = np.random.randn(200)
        sym = entropy_max_symbolize(x, n_bins=6)
        assert len(sym) == len(x)
        builder, sax_sym = sax_transition_network(x, n_bins=5, word_size=5)
        assert builder.n_nodes > 0
        assert len(sax_sym) > 0
