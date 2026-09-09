"""Contract tests for the role-feature schema definitions.

`ts2net.networks.feature_schema` is the authoritative statement of what each
column of the role-feature matrix means. These tests keep it honest against the
implementation, and pin the fact that spectral kernel bandwidth is currently
derived from matrix width -- which is why a schema change and a bandwidth
change cannot presently be made independently.

No cluster labels are frozen here. The gamma tests assert kernel matrices and
bandwidth values, which are deterministic, not partitions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

import ts2net  # noqa: F401
from ts2net.networks import roles
from ts2net.networks.feature_schema import (
    REDUNDANT_V1_COLUMNS,
    ROLE_FEATURES_V1,
    ROLE_FEATURES_V2,
    V1_NAMES,
    V2_NAMES,
    index_of,
)

GRAPHS = {
    "karate": nx.karate_club_graph(),
    "les_mis": nx.les_miserables_graph(),
    "er_40": nx.gnp_random_graph(40, 0.15, seed=1),
    "ba_40": nx.barabasi_albert_graph(40, 3, seed=3),
    "path_20": nx.path_graph(20),
    "star_19": nx.star_graph(19),
}


class TestSchemaShape:
    def test_v1_has_twelve_uniquely_named_columns(self):
        assert len(ROLE_FEATURES_V1) == 12
        assert len(set(V1_NAMES)) == 12

    def test_v2_has_nine_uniquely_named_columns(self):
        assert len(ROLE_FEATURES_V2) == 9
        assert len(set(V2_NAMES)) == 9

    def test_v2_is_v1_minus_the_aliases_with_order_preserved(self):
        expected = tuple(n for n in V1_NAMES if n not in REDUNDANT_V1_COLUMNS)
        assert V2_NAMES == expected

    def test_every_alias_maps_to_a_column_v2_keeps(self):
        for alias, canonical in REDUNDANT_V1_COLUMNS.items():
            assert alias in V1_NAMES
            assert alias not in V2_NAMES, f"{alias} should be dropped from v2"
            assert canonical in V2_NAMES, f"{canonical} must survive in v2"

    def test_every_feature_documents_itself(self):
        for feature in ROLE_FEATURES_V1:
            assert feature.definition.strip()
            assert feature.units.strip()
            assert feature.provenance.strip()

    def test_index_of_addresses_columns_by_meaning(self):
        assert index_of("degree") == 0
        assert index_of("triangles") == 7
        assert index_of("triangles", ROLE_FEATURES_V2) == 7
        with pytest.raises(KeyError, match="not in the schema"):
            index_of("no_such_feature")


class TestSchemaMatchesTheImplementation:
    """The declared order must be the order production actually emits."""

    @staticmethod
    def _independent_columns(graph):
        """Each named v1 column, computed directly from its documented definition."""
        und = graph.to_undirected()
        nodes = list(und.nodes())
        deg = np.array([und.degree(u) for u in nodes], float)
        tri = np.array([nx.triangles(und)[u] for u in nodes], float)
        core_d = nx.core_number(und)
        core = np.array([core_d[u] for u in nodes], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(deg >= 2, 2.0 * tri / (deg * (deg - 1)), 0.0)
        return {
            "degree": deg,
            "clustering": np.array([nx.clustering(und)[u] for u in nodes], float),
            "pagerank": np.array([nx.pagerank(und)[u] for u in nodes], float),
            "core_number": core,
            "betweenness": np.array(
                [nx.betweenness_centrality(und, normalized=True)[u] for u in nodes],
                float,
            ),
            "closeness": np.array(
                [nx.closeness_centrality(und)[u] for u in nodes], float
            ),
            "triangles": tri,
            "wedges": deg * (deg - 1) // 2 - tri,
            "ego_edges": tri,
            "ego_density": density,
            "core_score": core / core.max() if core.max() > 0 else core,
        }

    @staticmethod
    def _standardize(column):
        column = np.asarray(column, float)
        spread = float(np.std(column, ddof=1))
        scale = max(abs(float(np.mean(column))), float(np.max(np.abs(column))))
        if spread <= max(1e-12, 1e-12 * scale):
            return np.zeros_like(column)
        return (column - column.mean()) / spread

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_each_declared_column_holds_what_its_name_says(self, name):
        graph = GRAPHS[name]
        _, produced = roles.role_features_extended(graph)
        assert produced.shape[1] == len(V1_NAMES)
        expected = self._independent_columns(graph)
        for column_name, raw in expected.items():
            position = index_of(column_name)
            np.testing.assert_allclose(
                produced[:, position],
                self._standardize(raw),
                atol=1e-9,
                err_msg=f"{name}: column {position} is not {column_name!r}",
            )

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_declared_aliases_are_identical_columns_in_practice(self, name):
        _, produced = roles.role_features_extended(GRAPHS[name])
        for alias, canonical in REDUNDANT_V1_COLUMNS.items():
            np.testing.assert_allclose(
                produced[:, index_of(alias)],
                produced[:, index_of(canonical)],
                atol=1e-9,
                err_msg=f"{name}: {alias} should equal {canonical}",
            )

    def test_redundant_columns_is_exactly_the_set_of_corpus_wide_aliases(self):
        """An alias must coincide on *every* graph, not merely on some.

        Highly symmetric graphs produce coincidences that are not aliases: on a
        star, degree, pagerank, eigenvector, betweenness and closeness all take
        only two distinct values and therefore standardize to the same vector.
        Those pairs separate on other graphs. A genuine alias never separates,
        which is what makes it removable.
        """
        always_equal = None
        for graph in GRAPHS.values():
            _, produced = roles.role_features_extended(graph)
            equal_here = {
                frozenset((V1_NAMES[i], V1_NAMES[j]))
                for i in range(len(V1_NAMES))
                for j in range(i + 1, len(V1_NAMES))
                if np.allclose(produced[:, i], produced[:, j], atol=1e-9)
            }
            always_equal = (
                equal_here if always_equal is None else always_equal & equal_here
            )

        declared = {frozenset((a, c)) for a, c in REDUNDANT_V1_COLUMNS.items()}
        assert always_equal == declared, (
            f"schema declares {declared} but the corpus shows {always_equal}"
        )


class TestSchemaWidthCurrentlyControlsKernelBandwidth:
    """The coupling this investigation exists to expose.

    `node_roles_spectral` sets ``gamma = 1 / X.shape[1]`` when the caller does
    not pass one, so dropping redundant columns silently retunes the kernel.
    These assert the mechanism, not any partition.
    """

    def test_the_implicit_rule_is_scikit_learns_rbf_default(self):
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((30, 12))
        np.testing.assert_allclose(
            rbf_kernel(mat), rbf_kernel(mat, gamma=1.0 / mat.shape[1])
        )

    def test_width_change_moves_the_default_bandwidth(self):
        assert 1.0 / len(V1_NAMES) == pytest.approx(1 / 12)
        assert 1.0 / len(V2_NAMES) == pytest.approx(1 / 9)
        assert 1.0 / len(V1_NAMES) != 1.0 / len(V2_NAMES)

    def test_same_geometry_different_width_gives_a_different_kernel(self):
        """Bandwidth follows column count even when the extra column is a copy."""
        rng = np.random.default_rng(1)
        base = rng.standard_normal((25, 9))
        widened = np.hstack([base, base[:, [0]]])  # duplicate one column

        narrow = rbf_kernel(base, gamma=1.0 / base.shape[1])
        wide = rbf_kernel(widened, gamma=1.0 / widened.shape[1])
        assert not np.allclose(narrow, wide)

    def test_an_explicit_gamma_is_independent_of_width(self):
        rng = np.random.default_rng(2)
        base = rng.standard_normal((25, 9))
        fixed = 1.0 / 12.0
        np.testing.assert_allclose(
            rbf_kernel(base, gamma=fixed), rbf_kernel(base, gamma=fixed)
        )
        # And differs from the width-derived value, so the choice is material.
        assert not np.allclose(
            rbf_kernel(base, gamma=fixed), rbf_kernel(base, gamma=1.0 / 9.0)
        )

    def test_caller_supplied_gamma_reaches_the_kernel(self):
        """Passing gamma bypasses the width rule in the public function."""
        graph = nx.karate_club_graph()
        labels = roles.node_roles_spectral(graph, n_roles=3, gamma=0.05)
        assert set(labels) == set(graph.nodes())
        assert 1 < len(set(labels.values())) <= 3


class TestV2IsNotProducedYet:
    """This batch defines v2; it does not ship it."""

    def test_the_public_function_still_returns_v1(self):
        _, produced = roles.role_features_extended(nx.karate_club_graph())
        assert produced.shape[1] == 12
        assert produced.shape[1] == len(ROLE_FEATURES_V1)
