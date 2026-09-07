"""Tests for the single standardization boundary of the role-feature pipeline.

These pin the correctness contract established by
docs/audits/role_features_audit.md:

    numerically constant input column  ->  exactly zero standardized column

and the consequence that identical graph input yields an identical feature
matrix. They assert the standardizer's own behaviour on synthetic vectors,
which is fully deterministic, rather than depending on any particular ARPACK
random start.

No clustering assignment is frozen here.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
from ts2net.networks import roles
from ts2net.networks._standardize import (
    DEGENERACY_ATOL,
    DEGENERACY_RTOL,
    degenerate_columns,
    standardize,
)

NAMES = [
    "deg", "cc", "pr", "ev", "core", "btw", "clo",
    "tri", "wedges", "ego_edges", "ego_density", "core_score",
]


class TestDegeneracyCriterion:
    """Scale-aware, not `std == 0`."""

    @pytest.mark.parametrize(
        "label, column",
        [
            ("exact constant", np.full(20, 5.0)),
            ("zero-valued constant", np.zeros(20)),
            (
                "1.0 +/- 1e-15",
                1.0 + 1e-15 * np.array([(-1) ** i for i in range(20)], dtype=float),
            ),
            (
                "large constant +/- float error",
                1e6 + 1e-9 * np.array([(-1) ** i for i in range(20)], dtype=float),
            ),
            ("noise around zero", 1e-16 * np.linspace(-1.0, 1.0, 20)),
            ("relative spread 1e-15", 0.223607 * (1 + 1e-15 * np.arange(20))),
        ],
    )
    def test_numerically_constant_columns_are_degenerate(self, label, column):
        assert degenerate_columns(column.reshape(-1, 1))[0], label
        np.testing.assert_array_equal(
            standardize(column.reshape(-1, 1)), np.zeros((len(column), 1))
        )

    @pytest.mark.parametrize(
        "label, column, mean_tol",
        [
            ("unit-scale variation", np.linspace(0.0, 1.0, 20), 1e-9),
            # Variation of 1e-9 riding on an offset of 1.0 is only representable
            # to ~7 significant figures in float64: each input already carries
            # an absolute error near 1e-16, which is 1e-7 relative to the
            # spread. Centring cannot recover what the input never held, so the
            # standardized mean lands near 1e-6 rather than at machine zero.
            # This is a property of the input, not of the standardizer, and it
            # marks the practical floor of the degeneracy tolerance.
            ("small but real, near 1.0", 1.0 + np.linspace(0.0, 1e-9, 20), 1e-5),
            ("small but real, near 0", np.linspace(0.0, 1e-6, 20), 1e-9),
            ("integer counts", np.array([0.0, 1, 1, 2, 3, 5, 8, 13, 21, 34]), 1e-9),
            ("two distinct values", np.array([0.0] * 10 + [1.0] * 10), 1e-9),
        ],
    )
    def test_real_variation_is_not_zeroed(self, label, column, mean_tol):
        assert not degenerate_columns(column.reshape(-1, 1))[0], label
        out = standardize(column.reshape(-1, 1))
        assert out.std(ddof=1) == pytest.approx(1.0, abs=1e-9), label
        assert abs(out.mean()) < mean_tol, label

    def test_the_criterion_is_relative_not_absolute(self):
        """The same absolute spread is noise at 1e6 and signal at 1e-3."""
        spread = 1e-9 * np.array([(-1) ** i for i in range(20)], dtype=float)
        assert degenerate_columns((1e6 + spread).reshape(-1, 1))[0]
        assert not degenerate_columns((1e-3 + spread).reshape(-1, 1))[0]

    def test_tolerances_are_documented_constants(self):
        assert DEGENERACY_RTOL == 1e-12
        assert DEGENERACY_ATOL == 1e-12


class TestStandardizerCannotAmplify:
    """The defect the audit found, asserted directly on the standardizer."""

    def test_a_1e15_residue_becomes_exactly_zero_not_unit_variance(self):
        residue = 0.223607 + 1e-15 * np.random.default_rng(0).standard_normal(20)
        assert residue.std(ddof=1) < 1e-14
        out = standardize(residue.reshape(-1, 1))
        np.testing.assert_array_equal(out, np.zeros((20, 1)))
        assert out.std(ddof=1) == 0.0

    def test_applying_it_twice_changes_nothing(self):
        rng = np.random.default_rng(1)
        mat = np.column_stack(
            [rng.standard_normal(30), np.full(30, 7.0), rng.uniform(size=30) * 100]
        )
        once = standardize(mat)
        np.testing.assert_allclose(standardize(once), once, atol=1e-12)

    def test_mixed_matrix_zeroes_only_the_degenerate_column(self):
        rng = np.random.default_rng(2)
        mat = np.column_stack([rng.standard_normal(25), np.full(25, 3.0)])
        out = standardize(mat)
        assert out[:, 0].std(ddof=1) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_array_equal(out[:, 1], np.zeros(25))

    def test_fewer_than_two_rows_has_no_definable_variation(self):
        np.testing.assert_array_equal(standardize(np.array([[1.0, 2.0]])), [[0.0, 0.0]])

    def test_rejects_non_matrix_input(self):
        with pytest.raises(ValueError, match="2-D"):
            standardize(np.arange(5.0))


class TestRegularGraphsProduceZeroColumns:
    """Families the characterization showed exposing the defect."""

    @pytest.mark.parametrize(
        "name, graph, expect_zero",
        [
            ("cycle_20", nx.cycle_graph(20), set(NAMES)),
            ("complete_12", nx.complete_graph(12), set(NAMES)),
            (
                "path_20",
                nx.path_graph(20),
                {"cc", "core", "tri", "ego_edges", "ego_density", "core_score"},
            ),
            (
                "star_19",
                nx.star_graph(19),
                {"cc", "core", "tri", "ego_edges", "ego_density", "core_score"},
            ),
            (
                "bipartite_5_7",
                nx.complete_bipartite_graph(5, 7),
                {"cc", "core", "tri", "ego_edges", "ego_density", "core_score"},
            ),
            ("ba_40", nx.barabasi_albert_graph(40, 3, seed=3), {"core", "core_score"}),
            (
                "ws_40",
                nx.watts_strogatz_graph(40, 4, 0.1, seed=2),
                {"core", "core_score"},
            ),
        ],
    )
    def test_degenerate_columns_are_exactly_zero(self, name, graph, expect_zero):
        _, feats = roles.role_features_extended(graph)
        zeroed = {NAMES[i] for i in range(12) if np.all(feats[:, i] == 0.0)}
        assert expect_zero <= zeroed, f"{name}: expected zeroed {expect_zero - zeroed}"

    @pytest.mark.parametrize(
        "name, graph",
        [
            ("cycle_20", nx.cycle_graph(20)),
            ("complete_12", nx.complete_graph(12)),
            ("karate", nx.karate_club_graph()),
            ("er_40", nx.gnp_random_graph(40, 0.15, seed=1)),
        ],
    )
    def test_repeated_calls_agree(self, name, graph):
        """Integration counterpart to the standardizer tests above.

        Before the fix, cycle_20 and complete_12 differed by ~3 between calls
        because amplified ARPACK residue reached the output. They are now
        bit-stable; the others were always stable.
        """
        runs = [roles.role_features_extended(graph)[1] for _ in range(4)]
        for other in runs[1:]:
            np.testing.assert_allclose(runs[0], other, atol=1e-12)


class TestReconstructionInvariant:
    """Independently rebuild the output from raw columns and the public rule.

    This deliberately does not import `standardize`: it re-implements the
    documented rule with a different formulation (an explicit per-column loop
    and its own degeneracy test), so the test cannot pass by repeating a bug in
    the production implementation.
    """

    @staticmethod
    def _independent_standardize(mat):
        out = np.zeros_like(mat, dtype=float)
        n_rows = mat.shape[0]
        for j in range(mat.shape[1]):
            column = mat[:, j].astype(float)
            if n_rows < 2:
                continue
            # Documented rule: degenerate when the spread is at most
            # max(atol, rtol * scale), scale = max(|mean|, max|x|).
            spread = float(np.std(column, ddof=1))
            scale = max(abs(float(np.mean(column))), float(np.max(np.abs(column))))
            if spread <= max(1e-12, 1e-12 * scale):
                continue
            centred = column - np.mean(column)
            out[:, j] = centred / np.sqrt(np.sum(centred**2) / (n_rows - 1))
        return out

    @staticmethod
    def _raw_columns(graph):
        """The twelve raw columns, mirroring role_features_extended's assembly."""
        from ts2net.networks.communities import _role_features_basic

        und = graph.to_undirected()
        nodes, basic = _role_features_basic(und)
        nodes = list(nodes)
        motif = roles._motif_features(und, nodes)
        ego = roles._ego_edges_per_node(und).astype(float).reshape(-1, 1)
        density = roles._egonet_density(und, nodes).reshape(-1, 1)
        core = roles._core_periphery_scores(und, nodes).reshape(-1, 1)
        return np.hstack([basic, motif, ego, density, core])

    @pytest.mark.parametrize(
        "name, graph",
        [
            ("karate", nx.karate_club_graph()),
            ("les_mis", nx.les_miserables_graph()),
            ("er_40", nx.gnp_random_graph(40, 0.15, seed=1)),
            ("ba_40", nx.barabasi_albert_graph(40, 3, seed=3)),
            ("path_20", nx.path_graph(20)),
            ("star_19", nx.star_graph(19)),
            ("grid_5x5", nx.grid_2d_graph(5, 5)),
            ("barbell_8_3", nx.barbell_graph(8, 3)),
        ],
    )
    def test_reconstruction_matches_production(self, name, graph):
        _, produced = roles.role_features_extended(graph)
        rebuilt = self._independent_standardize(self._raw_columns(graph))
        np.testing.assert_allclose(produced, rebuilt, atol=1e-9)

    def test_basic_features_are_returned_raw(self):
        """Ownership check: the builder must not standardize."""
        from ts2net.networks.communities import _role_features_basic

        _, basic = _role_features_basic(nx.karate_club_graph())
        degrees = basic[:, 0]
        assert degrees.min() >= 1.0, "degree column should be raw counts"
        assert not np.isclose(degrees.std(ddof=1), 1.0), "must not be standardized"
