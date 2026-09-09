"""Tests for the RBF bandwidth policies of `node_roles_spectral`.

Bandwidth is a named policy, not a consequence of matrix shape. Two are
available: the fixed :data:`roles.DEFAULT_SPECTRAL_GAMMA` (the default) and the
data-driven median heuristic (opt-in via ``gamma="median"``).

These assert bandwidth values and kernel behaviour, which are deterministic.
No cluster labels are frozen.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401
from ts2net.networks import roles


class TestMedianHeuristicValue:
    def test_matches_the_definition_on_a_hand_computable_set(self):
        """gamma = 1 / (2 * median of squared pairwise distances)."""
        points = np.array([[0.0], [1.0], [3.0]])
        # squared distances: 1, 9, 4 -> median 4
        assert roles.median_heuristic_gamma(points) == pytest.approx(1.0 / 8.0)

    def test_scales_as_the_inverse_square_of_the_data(self):
        rng = np.random.default_rng(0)
        points = rng.standard_normal((40, 3))
        base = roles.median_heuristic_gamma(points)
        for factor in (2.0, 10.0):
            scaled = roles.median_heuristic_gamma(points * factor)
            assert scaled == pytest.approx(base / factor**2, rel=1e-9)

    def test_is_deterministic(self):
        rng = np.random.default_rng(1)
        points = rng.standard_normal((30, 4))
        assert roles.median_heuristic_gamma(points) == roles.median_heuristic_gamma(
            points
        )

    def test_rejects_non_matrix_input(self):
        with pytest.raises(ValueError, match="2-D"):
            roles.median_heuristic_gamma(np.arange(5.0))


class TestMedianHeuristicDegeneracy:
    """Its real failure mode: a median distance of zero.

    When more than half of all pairs coincide the median squared distance is
    zero and the implied bandwidth diverges. Coincident rows differ by
    floating-point residue, so the guard is scale aware rather than `== 0`.
    """

    def test_falls_back_when_most_pairs_coincide(self):
        # 8 identical rows and 2 distinct ones: most pairs are at distance 0.
        points = np.vstack([np.zeros((8, 2)), np.array([[5.0, 5.0], [7.0, 1.0]])])
        assert roles.median_heuristic_gamma(points) == roles.DEFAULT_SPECTRAL_GAMMA

    def test_falls_back_on_floating_point_residue_not_just_exact_zero(self):
        residue = 1e-18 * np.random.default_rng(2).standard_normal((10, 2))
        points = np.vstack([residue, np.array([[4.0, 4.0], [6.0, 2.0]])])
        upper = points[:, None, :] - points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", upper, upper)
        nonzero = squared[np.triu_indices_from(squared, k=1)]
        assert np.median(nonzero) > 0.0, "median is residue, not exactly zero"
        assert roles.median_heuristic_gamma(points) == roles.DEFAULT_SPECTRAL_GAMMA

    def test_falls_back_on_an_all_zero_matrix(self):
        assert roles.median_heuristic_gamma(np.zeros((12, 9))) == (
            roles.DEFAULT_SPECTRAL_GAMMA
        )

    def test_falls_back_with_fewer_than_two_rows(self):
        assert roles.median_heuristic_gamma(np.zeros((1, 4))) == (
            roles.DEFAULT_SPECTRAL_GAMMA
        )

    def test_the_fallback_value_is_configurable(self):
        assert roles.median_heuristic_gamma(np.zeros((5, 2)), fallback=0.5) == 0.5

    @pytest.mark.parametrize(
        "name, graph",
        [
            ("wheel_20", nx.wheel_graph(20)),
            ("star_19", nx.star_graph(19)),
            ("cycle_20", nx.cycle_graph(20)),
            ("complete_12", nx.complete_graph(12)),
        ],
    )
    def test_symmetric_graphs_degenerate_and_fall_back(self, name, graph):
        """A large automorphism group makes most node pairs coincide."""
        _, feats = roles.role_features_extended(graph)
        assert roles.median_heuristic_gamma(feats) == roles.DEFAULT_SPECTRAL_GAMMA

    @pytest.mark.parametrize(
        "name, graph",
        [
            ("karate", nx.karate_club_graph()),
            ("les_mis", nx.les_miserables_graph()),
            ("er_40", nx.gnp_random_graph(40, 0.15, seed=1)),
        ],
    )
    def test_graphs_with_distinguishable_nodes_get_a_real_bandwidth(self, name, graph):
        _, feats = roles.role_features_extended(graph)
        gamma = roles.median_heuristic_gamma(feats)
        assert gamma != roles.DEFAULT_SPECTRAL_GAMMA
        assert 0.0 < gamma < 1.0


class TestGammaPolicyResolution:
    @staticmethod
    def _features():
        return roles.role_features_extended(nx.karate_club_graph())[1]

    def test_none_selects_the_fixed_default(self):
        assert roles._resolve_gamma(self._features(), None) == (
            roles.DEFAULT_SPECTRAL_GAMMA
        )

    def test_median_selects_the_heuristic(self):
        feats = self._features()
        assert roles._resolve_gamma(feats, "median") == (
            roles.median_heuristic_gamma(feats)
        )

    def test_a_float_is_used_verbatim(self):
        assert roles._resolve_gamma(self._features(), 0.25) == 0.25

    def test_unknown_policy_and_non_positive_values_are_rejected(self):
        feats = self._features()
        with pytest.raises(ValueError, match="unknown gamma policy"):
            roles._resolve_gamma(feats, "silverman")
        with pytest.raises(ValueError, match="must be positive"):
            roles._resolve_gamma(feats, 0.0)
        with pytest.raises(ValueError, match="must be positive"):
            roles._resolve_gamma(feats, -1.0)


class TestDefaultIsUnchanged:
    """This slice adds a policy; it does not change what happens by default."""

    def test_the_default_remains_the_fixed_constant(self):
        assert roles.DEFAULT_SPECTRAL_GAMMA == pytest.approx(1 / 12)

    def test_omitting_gamma_matches_passing_the_constant(self):
        graph = nx.karate_club_graph()
        assert roles.node_roles_spectral(graph, n_roles=4, seed=3363) == (
            roles.node_roles_spectral(
                graph, n_roles=4, seed=3363, gamma=roles.DEFAULT_SPECTRAL_GAMMA
            )
        )

    def test_the_median_policy_is_usable_end_to_end(self):
        graph = nx.karate_club_graph()
        labels = roles.node_roles_spectral(graph, n_roles=4, gamma="median")
        assert set(labels) == set(graph.nodes())
        assert 1 < len(set(labels.values())) <= 4
