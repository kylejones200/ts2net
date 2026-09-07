"""Characterization tests for three known defects, captured before they are fixed.

Each test here asserts the behaviour the code has *today*, not the behaviour it
should have. They exist so the defects are demonstrable in version control
rather than described in prose. The commit that fixes each defect replaces the
corresponding test with one asserting correct behaviour.

Defects captured:

1. ``iaaft`` did not restore the original amplitude distribution, so it was not
   the iterative amplitude-adjusted Fourier transform its name claims. FIXED --
   replaced by tests/test_surrogate_correctness.py.
2. ``triangles_per_node`` returned twice the conventional per-node triangle
   count that ``networkx.triangles`` returns. FIXED -- replaced by
   tests/test_graph_metrics_parity.py.
3. ``ts2net.networks.roles`` advertises a Rust fast path that has never
   executed, because the symbols it imports do not exist in the extension.
   The module additionally cannot be imported at all: it imports
   ``ts2net.networks.utils``, which has never existed in any commit.
"""

import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net_rs


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
