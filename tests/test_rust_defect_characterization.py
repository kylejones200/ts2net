"""Characterization tests for three known defects, captured before they are fixed.

Each test here asserts the behaviour the code has *today*, not the behaviour it
should have. They exist so the defects are demonstrable in version control
rather than described in prose. The commit that fixes each defect replaces the
corresponding test with one asserting correct behaviour.

Defects captured:

1. ``iaaft`` does not restore the original amplitude distribution, so it is not
   the iterative amplitude-adjusted Fourier transform its name claims.
2. ``triangles_per_node`` returns twice the conventional per-node triangle
   count that ``networkx.triangles`` returns.
3. ``ts2net.networks.roles`` advertises a Rust fast path that has never
   executed, because the symbols it imports do not exist in the extension.
   The module additionally cannot be imported at all: it imports
   ``ts2net.networks.utils``, which has never existed in any commit.
"""

import networkx as nx
import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net_rs


def _signal(n=128):
    i = np.arange(n)
    return np.sin(i * 0.37) + 0.25 * np.cos(i * 1.1)


class TestIaaftDefect:
    """`iaaft` imposes the original's rank order, not its amplitudes."""

    def test_rust_iaaft_does_not_preserve_the_amplitude_distribution(self):
        x = _signal()
        surrogate = ts2net_rs.iaaft(x, 50, 3)
        assert not np.allclose(np.sort(x), np.sort(surrogate)), (
            "iaaft unexpectedly preserved the value distribution; if this "
            "fails the defect is fixed and this test should be replaced"
        )

    def test_rust_iaaft_preserves_the_rank_ordering_instead(self):
        x = _signal()
        surrogate = ts2net_rs.iaaft(x, 50, 3)
        np.testing.assert_array_equal(np.argsort(np.argsort(x)),
                                      np.argsort(np.argsort(surrogate)))

    def test_python_fallback_has_the_same_defect(self):
        from ts2net.stats.stats import iaaft as _dispatcher  # noqa: F401
        import ts2net.stats.stats as st

        # Force the NumPy fallback rather than the Rust path.
        rust, st._iaaft_rs = st._iaaft_rs, None
        try:
            x = _signal()
            surrogate = st.iaaft(x, iters=50, rng=3)
        finally:
            st._iaaft_rs = rust
        assert not np.allclose(np.sort(x), np.sort(surrogate))
        np.testing.assert_array_equal(np.argsort(np.argsort(x)),
                                      np.argsort(np.argsort(surrogate)))


class TestTrianglesDefect:
    """`triangles_per_node` double counts."""

    @pytest.mark.parametrize(
        "graph_edges, n",
        [
            ([(0, 1), (1, 2), (0, 2)], 3),                     # one triangle
            ([(0, 1), (1, 2), (0, 2), (2, 3), (3, 0)], 4),     # triangle + square
        ],
    )
    def test_rust_returns_twice_the_networkx_count(self, graph_edges, n):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(graph_edges)
        expected = np.array([nx.triangles(G)[u] for u in range(n)], dtype=np.int64)

        edges = np.asarray(graph_edges, dtype=np.uint64)
        got = np.asarray(ts2net_rs.triangles_per_node(n, edges), dtype=np.int64)

        np.testing.assert_array_equal(got, 2 * expected)


class TestRolesRustPathNeverRuns:
    """roles.py imports three symbols the extension does not export."""

    @pytest.mark.parametrize(
        "symbol", ["node_triangles", "ego_edge_counts", "core_numbers"]
    )
    def test_symbol_is_absent_from_the_extension(self, symbol):
        assert not hasattr(ts2net_rs, symbol)

    def test_the_module_cannot_be_imported_at_all(self):
        # The Rust fast path is unreachable for a more basic reason than the
        # missing symbols: importing the module raises. `ts2net.networks.utils`
        # is not present in the tree and never has been.
        import importlib

        with pytest.raises(ModuleNotFoundError, match=r"ts2net\.networks\.utils"):
            importlib.import_module("ts2net.networks.roles")

    def test_nothing_in_the_package_imports_it(self):
        import ts2net.networks as networks

        assert "roles" not in getattr(networks, "__all__", [])
        assert not hasattr(networks, "roles")
