"""Parity tests for ts2net.networks.roles and its new Rust primitives.

Until now `roles.py` could not be imported at all (it imported the nonexistent
``ts2net.networks.utils``) and the Rust fast path it advertised had never run,
because ``node_triangles``, ``ego_edge_counts`` and ``core_numbers`` were not
exported by the extension. Two of those are now implemented in Rust and the
third is served by the existing ``triangles_per_node``.

Every test that compares the two paths runs the *same* input through the Rust
implementation and the networkx fallback and requires identical results.
"""

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net_rs
from ts2net.networks import roles


def _edges_for_rust(G):
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    edges = np.empty((G.number_of_edges(), 2), dtype=np.uint64)
    for k, (u, v) in enumerate(G.edges()):
        edges[k, 0] = idx[u]
        edges[k, 1] = idx[v]
    return len(nodes), edges, nodes


@pytest.fixture
def networkx_fallback():
    """Force roles.py onto its networkx path for the duration of a test."""
    saved = (roles._tri_rs, roles._ego_rs, roles._core_rs)
    roles._tri_rs = None
    roles._ego_rs = None
    roles._core_rs = None
    try:
        yield roles
    finally:
        roles._tri_rs, roles._ego_rs, roles._core_rs = saved


GRAPHS = {
    "triangle": nx.Graph([(0, 1), (1, 2), (0, 2)]),
    "path": nx.Graph([(0, 1), (1, 2), (2, 3)]),
    "star": nx.star_graph(6),
    "k4_with_pendant": nx.Graph(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4)]
    ),
    "karate": nx.karate_club_graph(),
    "gnp_25": nx.gnp_random_graph(25, 0.25, seed=1),
    "gnp_40_sparse": nx.gnp_random_graph(40, 0.08, seed=2),
    "disconnected": nx.disjoint_union(nx.complete_graph(4), nx.path_graph(3)),
}


class TestExtensionExportsTheSymbols:
    """The inverse of the old characterization test."""

    @pytest.mark.parametrize(
        "symbol", ["triangles_per_node", "ego_edge_counts", "core_numbers"]
    )
    def test_symbol_is_exported(self, symbol):
        assert hasattr(ts2net_rs, symbol)

    def test_roles_bound_the_rust_path(self):
        # If a future build drops one of these bindings, importing roles.py
        # now raises AttributeError instead of silently using networkx.
        assert roles._tri_rs is not None
        assert roles._ego_rs is not None
        assert roles._core_rs is not None


class TestEgoEdgeCountsMatchNetworkX:
    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_against_networkx_subgraph_edge_count(self, name):
        G = GRAPHS[name]
        n, edges, nodes = _edges_for_rust(G)
        got = np.asarray(ts2net_rs.ego_edge_counts(n, edges), dtype=np.int64)
        expected = np.array(
            [G.subgraph(list(G.neighbors(u))).number_of_edges() for u in nodes],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_equals_triangles_per_node(self, name):
        # An edge between two neighbours of u is a triangle through u.
        G = GRAPHS[name]
        n, edges, _ = _edges_for_rust(G)
        np.testing.assert_array_equal(
            np.asarray(ts2net_rs.ego_edge_counts(n, edges)),
            np.asarray(ts2net_rs.triangles_per_node(n, edges)),
        )


class TestCoreNumbersMatchNetworkX:
    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_against_networkx_core_number(self, name):
        G = GRAPHS[name]
        n, edges, nodes = _edges_for_rust(G)
        got = np.asarray(ts2net_rs.core_numbers(n, edges), dtype=np.int64)
        core = nx.core_number(G)
        expected = np.array([core[u] for u in nodes], dtype=np.int64)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_on_random_graphs(self, seed):
        G = nx.gnp_random_graph(30, 0.18, seed=seed)
        n, edges, nodes = _edges_for_rust(G)
        got = np.asarray(ts2net_rs.core_numbers(n, edges), dtype=np.int64)
        core = nx.core_number(G)
        np.testing.assert_array_equal(got, [core[u] for u in nodes])


class TestRustPathMatchesFallback:
    """The two code paths inside roles.py must be interchangeable."""

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_triangles(self, name, networkx_fallback):
        G = GRAPHS[name]
        with_fallback = roles._triangles_per_node(G)
        roles._tri_rs = ts2net_rs.triangles_per_node
        np.testing.assert_array_equal(roles._triangles_per_node(G), with_fallback)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_ego_edges(self, name, networkx_fallback):
        G = GRAPHS[name]
        with_fallback = roles._ego_edges_per_node(G)
        roles._ego_rs = ts2net_rs.ego_edge_counts
        np.testing.assert_array_equal(roles._ego_edges_per_node(G), with_fallback)

    @pytest.mark.parametrize("name", sorted(GRAPHS))
    def test_core_numbers(self, name, networkx_fallback):
        G = GRAPHS[name]
        with_fallback = roles._core_number(G)
        roles._core_rs = ts2net_rs.core_numbers
        np.testing.assert_array_equal(roles._core_number(G), with_fallback)

    def test_feature_matrix_is_identical_either_way(self, networkx_fallback):
        G = nx.karate_club_graph()
        nodes_fb, X_fb = roles.role_features_extended(G)
        roles._tri_rs = ts2net_rs.triangles_per_node
        roles._ego_rs = ts2net_rs.ego_edge_counts
        roles._core_rs = ts2net_rs.core_numbers
        nodes_rs, X_rs = roles.role_features_extended(G)
        assert nodes_fb == nodes_rs
        np.testing.assert_allclose(X_fb, X_rs, atol=1e-12)


class TestNonIntegerNodeLabels:
    """`_edges_array` remaps labels to indices; results stay in node order."""

    def test_string_labelled_graph(self):
        G = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])
        tri = roles._triangles_per_node(G)
        core = roles._core_number(G)
        nodes = list(G.nodes())
        nx_tri = nx.triangles(G)
        nx_core = nx.core_number(G)
        np.testing.assert_array_equal(tri, [nx_tri[u] for u in nodes])
        np.testing.assert_array_equal(core, [nx_core[u] for u in nodes])


class TestRolesArePubliclyUsable:
    def test_exported_from_the_networks_package(self):
        from ts2net.networks import (
            node_roles_kmeans,
            node_roles_spectral,
            role_features_extended,
        )

        assert all(
            callable(f)
            for f in (node_roles_kmeans, node_roles_spectral, role_features_extended)
        )

    def test_clustering_runs_end_to_end(self):
        G = nx.karate_club_graph()
        labels = roles.node_roles_kmeans(G, n_roles=4, seed=3363)
        assert set(labels) == set(G.nodes())
        assert 1 < len(set(labels.values())) <= 4
