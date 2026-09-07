"""Parity tests for the Rust graph metrics against NetworkX.

`triangles_per_node` previously returned twice the conventional count. These
tests replace the `TestTrianglesDefect` characterization tests and pin the
Rust results to `networkx` on hand-computable graphs and on random ones.
"""

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net_rs
from ts2net.core.core_rust import triangles_per_node


def _edges_for_rust(G, n):
    """Edge list in the [m, 2] uint64 form the binding expects."""
    edges = np.empty((G.number_of_edges(), 2), dtype=np.uint64)
    for k, (u, v) in enumerate(G.edges()):
        edges[k, 0] = u
        edges[k, 1] = v
    return edges


def _networkx_triangles(G, n):
    tri = nx.triangles(G)
    return np.array([tri[u] for u in range(n)], dtype=np.int64)


HAND_COMPUTED = {
    # name: (n, edges, expected per-node triangle count)
    "single_triangle": (3, [(0, 1), (1, 2), (0, 2)], [1, 1, 1]),
    "two_triangles_sharing_an_edge": (
        4,
        [(0, 1), (1, 2), (0, 2), (2, 3), (3, 0)],
        [2, 1, 2, 1],
    ),
    "four_clique": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [3, 3, 3, 3]),
    "path": (3, [(0, 1), (1, 2)], [0, 0, 0]),
    "star": (4, [(0, 1), (0, 2), (0, 3)], [0, 0, 0, 0]),
    "isolated_nodes": (3, [], [0, 0, 0]),
}


class TestTrianglesMatchHandComputation:
    @pytest.mark.parametrize("name", sorted(HAND_COMPUTED))
    def test_hand_computed_graph(self, name):
        n, edge_list, expected = HAND_COMPUTED[name]
        edges = np.asarray(edge_list, dtype=np.uint64).reshape(-1, 2)
        got = np.asarray(ts2net_rs.triangles_per_node(n, edges), dtype=np.int64)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("name", sorted(HAND_COMPUTED))
    def test_hand_computed_graph_also_matches_networkx(self, name):
        n, edge_list, expected = HAND_COMPUTED[name]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edge_list)
        np.testing.assert_array_equal(_networkx_triangles(G, n), expected)


class TestTrianglesMatchNetworkX:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("n,p", [(12, 0.3), (25, 0.2), (40, 0.12)])
    def test_random_graphs(self, n, p, seed):
        G = nx.gnp_random_graph(n, p, seed=seed)
        got = np.asarray(
            ts2net_rs.triangles_per_node(n, _edges_for_rust(G, n)), dtype=np.int64
        )
        np.testing.assert_array_equal(got, _networkx_triangles(G, n))

    def test_the_python_wrapper_agrees_too(self):
        G = nx.gnp_random_graph(20, 0.25, seed=7)
        got = triangles_per_node(20, _edges_for_rust(G, 20))
        np.testing.assert_array_equal(got, _networkx_triangles(G, 20))

    def test_duplicate_and_reversed_edges_are_deduplicated(self):
        edges = np.asarray(
            [(0, 1), (1, 0), (0, 1), (1, 2), (0, 2)], dtype=np.uint64
        )
        got = np.asarray(ts2net_rs.triangles_per_node(3, edges), dtype=np.int64)
        np.testing.assert_array_equal(got, [1, 1, 1])


class TestClusteringMatchesNetworkX:
    """clustering_avg was untouched; pin it so the triangle change is isolated."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_average_clustering(self, seed):
        G = nx.gnp_random_graph(25, 0.25, seed=seed)
        got = ts2net_rs.clustering_avg(25, _edges_for_rust(G, 25))
        # ts2net averages only over nodes of degree >= 2; networkx counts
        # degree < 2 nodes as 0. Compare on the same subset.
        local = nx.clustering(G)
        eligible = [u for u in G.nodes() if G.degree(u) >= 2]
        expected = float(np.mean([local[u] for u in eligible]))
        assert got == pytest.approx(expected, abs=1e-12)
