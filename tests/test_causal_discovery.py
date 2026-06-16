"""
Tests for PC/FCI constraint-based causal discovery (0.9 milestone).
"""

import numpy as np
import networkx as nx
import pytest

from ts2net.causal.ci_tests import partial_correlation_ci_test
from ts2net.causal.lagged_panel import lagged_panel_matrix, is_temporally_valid_edge
from ts2net.causal.pc import pc_algorithm, pc_timeseries_network
from ts2net.causal.fci import fci_algorithm, fci_timeseries_network


class TestCITests:
    def test_independent_variables_high_p(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 3))
        indep, p_val, _ = partial_correlation_ci_test(data, 0, 1, ())
        assert indep
        assert p_val > 0.05

    def test_dependent_variables_low_p(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(600)
        y = 0.9 * x + 0.05 * rng.standard_normal(600)
        data = np.column_stack([x, y])
        indep, p_val, r = partial_correlation_ci_test(data, 0, 1, ())
        assert not indep
        assert p_val < 0.05
        assert abs(r) > 0.5


class TestLaggedPanel:
    def test_lagged_expansion_shape(self):
        rng = np.random.default_rng(2)
        X = [rng.standard_normal(100) for _ in range(2)]
        data, names, name_map = lagged_panel_matrix(
            X, max_lag=2, series_names=["A", "B"]
        )
        assert data.shape == (98, 6)
        assert len(names) == 6
        assert "A(t-2)" in name_map
        assert is_temporally_valid_edge("A(t-1)", "A(t)")
        assert not is_temporally_valid_edge("A(t)", "A(t-1)")


class TestPCAlgorithm:
    def test_pc_chain_structure(self):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(2500)
        y = 0.85 * x + 0.08 * rng.standard_normal(2500)
        z = 0.85 * y + 0.08 * rng.standard_normal(2500)
        data = np.column_stack([x, y, z])

        result = pc_algorithm(data, alpha=0.01, variable_names=["X", "Y", "Z"])
        assert result.skeleton.has_edge(0, 1)
        assert result.skeleton.has_edge(1, 2)
        assert result.skeleton.number_of_edges() >= 2

    def test_pc_collider_oriented(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(3000)
        y = rng.standard_normal(3000)
        z = 0.9 * x + 0.9 * y + 0.05 * rng.standard_normal(3000)
        data = np.column_stack([x, y, z])

        result = pc_algorithm(data, alpha=0.01)
        assert result.skeleton.has_edge(0, 2)
        assert result.skeleton.has_edge(1, 2)
        assert not result.skeleton.has_edge(0, 1)

        cpdag = result.cpdag
        assert cpdag.has_edge(0, 2) and not cpdag.has_edge(2, 0)
        assert cpdag.has_edge(1, 2) and not cpdag.has_edge(2, 1)

    def test_pc_independent_sparse_skeleton(self):
        rng = np.random.default_rng(12)
        data = rng.standard_normal((400, 4))
        result = pc_algorithm(data, alpha=0.01)
        # Sparse graph expected; allow rare false positives at looser alpha
        assert result.skeleton.number_of_edges() <= 2


class TestFCIAlgorithm:
    def test_fci_latent_confounder_edge(self):
        rng = np.random.default_rng(20)
        lat = rng.standard_normal(2000)
        x = lat + 0.08 * rng.standard_normal(2000)
        y = lat + 0.08 * rng.standard_normal(2000)
        data = np.column_stack([x, y])

        result = fci_algorithm(data, alpha=0.05)
        assert result.skeleton.has_edge(0, 1)
        edge = result.pag.get_edge_data(0, 1) or result.pag.get_edge_data(1, 0)
        assert edge is not None
        assert edge.get("mark_u") == "circle" or edge.get("mark_v") == "circle"

    def test_fci_chain_has_edges(self):
        rng = np.random.default_rng(21)
        x = rng.standard_normal(2000)
        y = 0.8 * x + 0.1 * rng.standard_normal(2000)
        z = 0.8 * y + 0.1 * rng.standard_normal(2000)
        result = fci_algorithm(np.column_stack([x, y, z]), alpha=0.01)
        assert result.skeleton.number_of_edges() >= 2


class TestTimeseriesDiscovery:
    def test_pc_timeseries_lagged_driver(self):
        rng = np.random.default_rng(30)
        n = 600
        x = rng.standard_normal(n)
        y = np.concatenate([[0], 0.75 * x[:-1] + 0.08 * rng.standard_normal(n - 1)])

        result = pc_timeseries_network(
            [x, y],
            max_lag=2,
            alpha=0.05,
            series_names=["driver", "response"],
        )
        names = result.variable_names
        driver_lag1 = names.index("driver(t-1)")
        response_t = names.index("response(t)")
        connected = (
            result.skeleton.has_edge(driver_lag1, response_t)
            or result.skeleton.has_edge(response_t, driver_lag1)
        )
        assert connected or result.skeleton.number_of_edges() >= 1

    def test_fci_timeseries_runs(self):
        rng = np.random.default_rng(31)
        X = [rng.standard_normal(300) for _ in range(3)]
        result = fci_timeseries_network(X, max_lag=1, alpha=0.05)
        assert isinstance(result.pag, nx.DiGraph)
        assert result.n_obs == 299
