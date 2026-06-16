"""
Tests for dynamic network analytics (horizon 0.8).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ts2net.dynamic import (
    DynamicWorkflowSpec,
    community_labels,
    detect_regime_changes,
    edge_transition_anomalies,
    node_role_evolution,
    node_roles,
    run_dynamic_analysis,
    track_communities,
    window_anomaly_scores,
)


class TestRegimeDetection:
    def test_detects_shift(self):
        values = np.concatenate([np.ones(30), np.ones(30) * 5.0])
        result = detect_regime_changes(values, threshold=2.0)
        assert len(result["break_indices"]) >= 1

    def test_short_series(self):
        result = detect_regime_changes(np.array([1.0, 2.0]))
        assert len(result["break_indices"]) == 0


class TestAnomalyScores:
    def test_window_anomaly_spike(self):
        stats = {
            "avg_degree": np.concatenate([np.ones(20), [10.0], np.ones(19)]),
            "n_edges": np.ones(40) * 50,
        }
        scores = window_anomaly_scores(stats)
        assert scores[20] == max(scores)

    def test_transition_anomalies(self):
        births = np.array([1.0, 1.0, 20.0, 1.0])
        deaths = np.array([1.0, 1.0, 1.0, 1.0])
        scores = edge_transition_anomalies(births, deaths)
        assert scores[2] == max(scores)


class TestRolesAndCommunities:
    def test_node_roles_star(self):
        G = nx.star_graph(5)
        roles = node_roles(G)
        assert roles[0] == "hub"

    def test_role_evolution(self):
        G1 = nx.path_graph(4)
        G2 = nx.star_graph(3)
        traj = node_role_evolution([G1, G2])
        assert len(traj[0]) == 2

    def test_community_tracking(self):
        G1 = nx.path_graph(6)
        G2 = nx.cycle_graph(6)
        out = track_communities([G1, G2])
        assert len(out["n_communities"]) == 2
        assert len(out["stability"]) == 1

    def test_community_labels_connected(self):
        graph = nx.star_graph(5)
        labels = community_labels(graph)
        assert len(labels) == graph.number_of_nodes()


class TestDynamicWorkflow:
    def test_run_dynamic_analysis(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(400)
        x[200:] += 2.5
        result = run_dynamic_analysis(
            x,
            DynamicWorkflowSpec(window=40, step=20, method="hvg"),
        )
        assert len(result.sequence.stats) > 0
        assert result.anomalies.shape[0] == len(result.sequence.stats)
        report = result.summary()
        assert "Dynamic network analysis" in report

    def test_kwargs_override(self):
        x = np.sin(np.linspace(0, 8 * np.pi, 300))
        result = run_dynamic_analysis(x, window=30, step=15, method="hvg")
        assert result.window == 30
        assert result.step == 15
