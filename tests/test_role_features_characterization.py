"""Mathematical contracts behind the role-feature schema.

These document why the v2 schema drops three columns. The identities are
properties of the underlying graph quantities and remain true whether or not
any column exposes them, so they are asserted against `ts2net_rs` and
`networkx` directly -- the level at which those quantities still live.

They do not freeze clustering assignments.

Two families of fact are pinned here:

1. Three of the twelve columns are exact aliases of three others. Each
   equivalence is an algebraic identity, verified on hand-built graphs that
   isolate structural regimes and on randomized graphs.
2. Standardization *was* applied twice -- once inside `_role_features_basic`
   and again over the concatenation -- which renormalized a numerically
   constant column to unit variance and made the output non-deterministic on
   vertex-transitive graphs. That defect is FIXED; see
   tests/test_feature_standardization.py for the current contract. The tests
   below retain the arithmetic that demonstrates why the fix was necessary.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net_rs
from ts2net.networks import roles


# Families chosen to expose different structural regimes, including graphs
# with many degree-1 nodes and with isolated (degree-0) nodes.
def _isolated_plus_triangle():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return graph


CORPUS = {
    "path_20": nx.path_graph(20),
    "cycle_20": nx.cycle_graph(20),
    "star_19": nx.star_graph(19),
    "wheel_20": nx.wheel_graph(20),
    "complete_12": nx.complete_graph(12),
    "bipartite_5_7": nx.complete_bipartite_graph(5, 7),
    "tree_2_4": nx.balanced_tree(2, 4),
    "grid_5x5": nx.grid_2d_graph(5, 5),
    "barbell_8_3": nx.barbell_graph(8, 3),
    "karate": nx.karate_club_graph(),
    "les_mis": nx.les_miserables_graph(),
    "isolated_plus_triangle": _isolated_plus_triangle(),
    "er_40": nx.gnp_random_graph(40, 0.15, seed=1),
    "ws_40": nx.watts_strogatz_graph(40, 4, 0.1, seed=2),
    "ba_40": nx.barabasi_albert_graph(40, 3, seed=3),
}

NAMES = pytest.mark.parametrize("name", sorted(CORPUS))


def _degrees(graph, nodes):
    return np.array([graph.degree(u) for u in nodes], dtype=float)


def _rust_edges(graph):
    """`(n, edges, nodes)` in the [m, 2] uint64 form the Rust bindings take."""
    und = graph.to_undirected()
    nodes = list(und.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    edges = np.empty((und.number_of_edges(), 2), dtype=np.uint64)
    for k, (u, v) in enumerate(und.edges()):
        edges[k, 0] = index[u]
        edges[k, 1] = index[v]
    return len(nodes), edges, nodes


class TestEgoEdgesEqualsTriangles:
    """`ego_edge_counts(u) == triangles_per_node(u)` for simple graphs.

    An edge between two neighbours of u closes a triangle through u, and every
    triangle through u contributes exactly one such edge. The map is a
    bijection, so the counts are equal for every node.
    """

    @NAMES
    def test_identity_holds(self, name):
        graph = CORPUS[name]
        n, edges, _ = _rust_edges(graph)
        np.testing.assert_array_equal(
            ts2net_rs.ego_edge_counts(n, edges), ts2net_rs.triangles_per_node(n, edges)
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_identity_holds_on_random_graphs(self, seed):
        graph = nx.gnp_random_graph(30, 0.2, seed=seed)
        n, edges, _ = _rust_edges(graph)
        np.testing.assert_array_equal(
            ts2net_rs.ego_edge_counts(n, edges), ts2net_rs.triangles_per_node(n, edges)
        )

    def test_both_are_zero_for_degree_zero_and_one_nodes(self):
        graph = _isolated_plus_triangle()
        n, edges, nodes = _rust_edges(graph)
        low = _degrees(graph, nodes) <= 1
        assert low.sum() >= 3, "fixture should contain isolated nodes"
        assert np.all(np.asarray(ts2net_rs.triangles_per_node(n, edges))[low] == 0)
        assert np.all(np.asarray(ts2net_rs.ego_edge_counts(n, edges))[low] == 0)


class TestEgonetDensityIsTheClusteringCoefficient:
    """`_egonet_density(u) == 2*T(u) / (k(u) * (k(u)-1))`, i.e. `nx.clustering`."""

    @NAMES
    def test_ego_density_closed_form_is_the_clustering_coefficient(self, name):
        graph = CORPUS[name]
        nodes = list(graph.nodes())
        deg = _degrees(graph, nodes)
        n, edges, _ = _rust_edges(graph)
        tri = np.asarray(ts2net_rs.triangles_per_node(n, edges), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ego_density = np.where(deg >= 2, 2.0 * tri / (deg * (deg - 1)), 0.0)
        clustering = np.array([nx.clustering(graph)[u] for u in nodes], dtype=float)
        np.testing.assert_allclose(ego_density, clustering, atol=1e-12)

    def test_degree_zero_and_one_are_defined_as_zero(self):
        """The denominator k(k-1) vanishes; the convention is 0 on both sides."""
        graph = _isolated_plus_triangle()
        nodes = list(graph.nodes())
        low = _degrees(graph, nodes) <= 1
        assert np.all([nx.clustering(graph)[u] == 0.0 for u in np.array(nodes)[low]])


class TestCoreScoreIsARescalingOfCore:
    """`core_score = core / max(core)`, which standardization then erases."""

    @NAMES
    def test_standardization_makes_core_score_identical_to_core(self, name):
        """z(a*x) == z(x) for a > 0, so the two columns coincided downstream."""
        graph = CORPUS[name]
        n, edges, _ = _rust_edges(graph)
        core = np.asarray(ts2net_rs.core_numbers(n, edges), dtype=float)
        score = core / core.max() if core.max() > 0 else core

        def z(x):
            return (x - x.mean()) / (x.std(ddof=1) + 1e-12)

        np.testing.assert_allclose(z(core), z(score), atol=1e-9)


class TestWedgesIsDeterminedByDegreeAndTriangles:
    @NAMES
    def test_closed_form(self, name):
        graph = CORPUS[name]
        nodes = list(graph.nodes())
        deg = _degrees(graph, nodes).astype(np.int64)
        tri = roles._triangles_per_node(graph)
        wedges = roles._motif_features(graph, nodes)[:, 1]
        np.testing.assert_array_equal(wedges, deg * (deg - 1) // 2 - tri)

    @NAMES
    def test_the_clamp_at_zero_never_binds(self, name):
        """T(u) <= C(k(u), 2) always, so `np.maximum(..., 0)` is a no-op."""
        graph = CORPUS[name]
        nodes = list(graph.nodes())
        deg = _degrees(graph, nodes).astype(np.int64)
        tri = roles._triangles_per_node(graph)
        assert np.all(deg * (deg - 1) // 2 - tri >= 0)


class TestFeatureMatrixRank:
    """Nine columns, and after the v2 change the rank can reach all nine."""

    @NAMES
    def test_rank_never_exceeds_the_column_count(self, name):
        _, feats = roles.role_features_extended(CORPUS[name])
        assert feats.shape[1] == 9
        assert np.linalg.matrix_rank(feats, tol=1e-9) <= 9

    @pytest.mark.parametrize("name", ["karate", "les_mis", "er_40"])
    def test_rank_is_exactly_nine_when_all_nine_signals_vary(self, name):
        graph = CORPUS[name]
        n, edges, _ = _rust_edges(graph)
        core = np.asarray(ts2net_rs.core_numbers(n, edges), dtype=float)
        assert core.std(ddof=1) > 1e-12, "fixture must have varying coreness"
        _, feats = roles.role_features_extended(graph)
        assert np.linalg.matrix_rank(feats, tol=1e-9) == 9

    @pytest.mark.parametrize("name", ["ba_40", "ws_40"])
    def test_coreness_is_constant_on_ba_and_ws_so_rank_drops(self, name):
        """Documented degeneracy, not a defect in the rank contract.

        Barabasi-Albert with fixed m, and Watts-Strogatz with fixed k, produce
        graphs in which every node has the same k-core number. Both `core` and
        `core_score` are then zero-variance, standardize to all-zero columns,
        and contribute nothing to any distance.
        """
        graph = CORPUS[name]
        n, edges, _ = _rust_edges(graph)
        core = np.asarray(ts2net_rs.core_numbers(n, edges), dtype=float)
        assert core.std(ddof=1) == 0.0
        _, feats = roles.role_features_extended(graph)
        assert np.linalg.matrix_rank(feats, tol=1e-9) == 8
        # Column 4 is core_number; v2 no longer carries its core_score alias.
        np.testing.assert_allclose(feats[:, 4], 0.0, atol=1e-9)


class TestWhyStandardizationHappensExactlyOnce:
    """The arithmetic that made double standardization a defect.

    Historical: `_role_features_basic` standardized its seven columns and
    `role_features_extended` standardized the concatenation again. The `+ 1e-12`
    guard in the first pass kept a near-constant column small but non-zero; the
    second pass saw a standard deviation far above the guard and rescaled that
    floating-point residue to unit variance.

    The pipeline now standardizes once, through
    `ts2net.networks._standardize.standardize`, which maps a degenerate column
    to exactly zero. These tests keep the old arithmetic in view so the reason
    for the single-boundary design stays legible; they operate on synthetic
    vectors with a local helper and do not exercise production code.
    """

    @staticmethod
    def _z(x):
        return (x - x.mean()) / (x.std(ddof=1) + 1e-12)

    def test_eigenvector_centrality_is_numerically_constant_on_a_cycle(self):
        graph = nx.cycle_graph(20)
        values = np.array(
            [nx.eigenvector_centrality_numpy(graph)[u] for u in graph.nodes()]
        )
        # Every node is equivalent under the automorphism group, so the true
        # centrality is uniform; what remains is float residue.
        assert values.std(ddof=1) < 1e-12

    def test_two_passes_rescale_float_residue_to_unit_variance(self):
        residue = np.array([1.0 + 1e-16 * s for s in range(20)])
        assert residue.std(ddof=1) < 1e-12

        once = self._z(residue)
        assert once.std(ddof=1) < 1e-2, "guard should keep one pass small"

        twice = self._z(once)
        assert twice.std(ddof=1) == pytest.approx(1.0, abs=1e-6), (
            "second pass renormalizes the residue to a full-amplitude feature"
        )

    def test_a_genuinely_varying_column_is_unaffected_by_the_second_pass(self):
        values = np.linspace(0.0, 5.0, 25)
        np.testing.assert_allclose(self._z(values), self._z(self._z(values)), atol=1e-9)
