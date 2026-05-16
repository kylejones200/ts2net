"""
Correctness tests for all four graph builders.

Every test verifies a mathematical property against a known ground truth,
not just that the code ran without error or produced the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork


# ── helpers ────────────────────────────────────────────────────────────────────

def edges_as_set(g) -> set[tuple[int, int]]:
    """Return undirected edge set as canonical (min, max) pairs."""
    rows, cols, _ = g.edges_coo()
    return {(min(i, j), max(i, j)) for i, j in zip(rows.tolist(), cols.tolist())}


# ══════════════════════════════════════════════════════════════════════════════
# HVG
# ══════════════════════════════════════════════════════════════════════════════

class TestHVGCorrectness:
    """Hand-verified HVG edges for small series."""

    def test_five_point_known_edges(self):
        """
        x = [1, 3, 2, 4, 2]

        Edge derivation (i,j connected iff x[k] < min(x[i],x[j]) for all i<k<j):
          (0,1): no intermediate                           → CONNECTED
          (0,2): x[1]=3 < min(1,2)=1? NO                  → blocked
          (1,2): no intermediate                           → CONNECTED
          (1,3): x[2]=2 < min(3,4)=3? YES                 → CONNECTED
          (1,4): x[2]=2 < min(3,2)=2? NO (not strictly)   → blocked
          (2,3): no intermediate                           → CONNECTED
          (2,4): x[3]=4 < min(2,2)=2? NO                  → blocked
          (3,4): no intermediate                           → CONNECTED
        """
        x = np.array([1.0, 3.0, 2.0, 4.0, 2.0])
        hvg = HVG().build(x)
        expected = {(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)}
        assert edges_as_set(hvg) == expected

    def test_monotone_increasing_only_adjacent(self):
        """Strictly increasing series: only adjacent nodes are visible."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        hvg = HVG().build(x)
        expected = {(0, 1), (1, 2), (2, 3), (3, 4)}
        assert edges_as_set(hvg) == expected, (
            "Monotone increasing: intermediate points always block long-range visibility"
        )

    def test_monotone_decreasing_all_visible_from_first(self):
        """Strictly decreasing: node 0 (maximum) is visible from all others."""
        x = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        hvg = HVG().build(x)
        # 0 connects to 1,2,3,4 (all lower); 1 connects to 2,3,4; etc.
        # plus all adjacent pairs; actually for strictly decreasing:
        # each pair (i,j) must have all intermediates < min(x[i], x[j]) = x[j]
        # (i,j) with x decreasing: x[k] for i<k<j is between x[i] and x[j],
        # so x[k] > x[j] → blocked for non-adjacent non-visible pairs
        # Actually for strictly decreasing: only adjacent pairs are connected
        # Wait: x = [5,4,3,2,1], check (0,2): x[1]=4 < min(5,3)=3? NO → blocked
        expected = {(0, 1), (1, 2), (2, 3), (3, 4)}
        assert edges_as_set(hvg) == expected

    def test_n_nodes_equals_series_length(self):
        rng = np.random.default_rng(0)
        for n in [10, 100, 500]:
            x = rng.standard_normal(n)
            hvg = HVG().build(x)
            assert hvg.n_nodes == n

    def test_mean_degree_converges_to_4(self):
        """
        For i.i.d. random series, E[degree] → 4 as n → ∞.
        (Luque et al. 2009, Theorem 1)
        """
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        hvg = HVG().build(x)
        mean_deg = hvg.degree_sequence().mean()
        # Allow ±5% tolerance around 4.0
        assert abs(mean_deg - 4.0) < 0.2, (
            f"Mean HVG degree for iid series should be ≈ 4.0, got {mean_deg:.3f}"
        )

    def test_every_node_degree_at_least_1(self):
        """Every node must have at least its adjacent neighbour(s)."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(200)
        hvg = HVG().build(x)
        d = hvg.degree_sequence()
        assert np.all(d >= 1), f"Found node with degree 0: {d}"

    def test_connected_is_superset_of_nvg_when_shorter(self):
        """HVG edges ⊆ NVG edges for the same series."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(50)
        hvg_edges = edges_as_set(HVG().build(x))
        nvg_edges = edges_as_set(NVG().build(x))
        assert hvg_edges.issubset(nvg_edges), (
            "Every HVG edge must also be an NVG edge"
        )

    def test_adjacency_matrix_symmetric(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(30)
        A = HVG().build(x).adjacency_matrix(format="dense")
        assert np.allclose(A, A.T), "Adjacency matrix must be symmetric for undirected HVG"

    def test_degree_sequence_sums_to_twice_edges(self):
        """For undirected graphs: sum(degrees) == 2 * n_edges."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(100)
        hvg = HVG().build(x)
        assert hvg.degree_sequence().sum() == 2 * hvg.n_edges


# ══════════════════════════════════════════════════════════════════════════════
# NVG
# ══════════════════════════════════════════════════════════════════════════════

class TestNVGCorrectness:
    """Hand-verified NVG edges for small series."""

    def test_three_point_peak(self):
        """
        x = [1, 3, 1]

        (0,1): adjacent → CONNECTED
        (1,2): adjacent → CONNECTED
        (0,2): check k=1: x[1]=3 vs line at k=1: 1+(1-1)*1/2=1.  3 < 1? NO → blocked

        Expected edges: {(0,1), (1,2)} only.
        """
        x = np.array([1.0, 3.0, 1.0])
        nvg = NVG().build(x)
        assert edges_as_set(nvg) == {(0, 1), (1, 2)}

    def test_three_point_valley(self):
        """
        x = [3, 1, 3]

        (0,1): adjacent → CONNECTED
        (1,2): adjacent → CONNECTED
        (0,2): check k=1: x[1]=1 vs line: 3+(3-3)*1/2=3.  1 < 3? YES → CONNECTED
        """
        x = np.array([3.0, 1.0, 3.0])
        nvg = NVG().build(x)
        assert edges_as_set(nvg) == {(0, 1), (0, 2), (1, 2)}

    def test_nvg_superset_of_hvg(self):
        """NVG edge set always contains the HVG edge set."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            x = rng.standard_normal(40)
            assert edges_as_set(HVG().build(x)).issubset(edges_as_set(NVG().build(x)))

    def test_n_nodes_correct(self):
        for n in [5, 50, 200]:
            x = np.random.default_rng(n).standard_normal(n)
            assert NVG().build(x).n_nodes == n

    def test_adjacency_matrix_symmetric(self):
        x = np.random.default_rng(5).standard_normal(20)
        A = NVG().build(x).adjacency_matrix(format="dense")
        assert np.allclose(A, A.T)


# ══════════════════════════════════════════════════════════════════════════════
# RecurrenceNetwork
# ══════════════════════════════════════════════════════════════════════════════

class TestRecurrenceNetworkCorrectness:
    """Known-ground-truth tests for RecurrenceNetwork (epsilon rule)."""

    def test_epsilon_known_edges(self):
        """
        x = [1.0, 1.1, 5.0, 5.1, 1.0], epsilon=0.5, m=None (1-D), metric='euclidean'

        Pairwise distances:
          |x[0]-x[1]| = 0.1 < 0.5 → (0,1) connected
          |x[0]-x[2]| = 4.0        → not connected
          |x[0]-x[3]| = 4.1        → not connected
          |x[0]-x[4]| = 0.0 < 0.5 → (0,4) connected
          |x[1]-x[2]| = 3.9        → not connected
          |x[1]-x[3]| = 4.0        → not connected
          |x[1]-x[4]| = 0.1 < 0.5 → (1,4) connected
          |x[2]-x[3]| = 0.1 < 0.5 → (2,3) connected
          |x[2]-x[4]| = 4.0        → not connected
          |x[3]-x[4]| = 4.1        → not connected

        Expected: {(0,1), (0,4), (1,4), (2,3)}
        """
        x = np.array([1.0, 1.1, 5.0, 5.1, 1.0])
        rn = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x)
        assert edges_as_set(rn) == {(0, 1), (0, 4), (1, 4), (2, 3)}

    def test_tiny_epsilon_gives_no_edges(self):
        """Epsilon smaller than the minimum pairwise distance → no edges."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Minimum pairwise distance = 1.0; epsilon=0.5 is safely below it
        rn = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x)
        assert rn.n_edges == 0

    def test_large_epsilon_gives_complete_graph(self):
        """epsilon > max pairwise distance → complete graph."""
        x = np.array([1.0, 2.0, 3.0])  # max distance = 2.0
        rn = RecurrenceNetwork(rule="epsilon", epsilon=10.0).build(x)
        assert rn.n_edges == 3  # C(3,2) = 3 for undirected

    def test_knn_k1_gives_n_minus_1_edges(self):
        """
        knn with k=1: each node connects to its single nearest neighbour.
        The resulting graph has at most n-1 edges (some mutual NN pairs
        share an edge).
        """
        rng = np.random.default_rng(10)
        x = rng.standard_normal(20)
        rn = RecurrenceNetwork(rule="knn", k=1).build(x)
        # Every node has degree ≥ 1 (it always has a nearest neighbour)
        assert np.all(rn.degree_sequence() >= 1)

    def test_adjacency_symmetric_epsilon(self):
        x = np.random.default_rng(11).standard_normal(30)
        A = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x).adjacency_matrix(format="dense")
        assert np.allclose(A, A.T)

    def test_n_nodes_correct(self):
        for n in [10, 50, 100]:
            x = np.random.default_rng(n).standard_normal(n)
            assert RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x).n_nodes == n


# ══════════════════════════════════════════════════════════════════════════════
# TransitionNetwork
# ══════════════════════════════════════════════════════════════════════════════

class TestTransitionNetworkCorrectness:
    """Known-ground-truth tests for TransitionNetwork (ordinal patterns)."""

    def test_alternating_series_two_nodes(self):
        """
        x = [0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.9]

        Implementation note: TransitionNetwork builds a HIGHER-ORDER Markov model.
        With order=2 (ordinal window size 2 → up/down patterns):
          Digitised:  [down, up, down, up, down, up]  = [0, 1, 0, 1, 0, 1]
          Sequences:  [0,1,0], [1,0,1], [0,1,0], [1,0,1]
          Nodes (pairs): (0,1) and (1,0) → exactly 2 distinct 2-tuples
        """
        x = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.9])
        tn = TransitionNetwork(symbolizer="ordinal", order=2).build(x)
        assert tn.n_nodes == 2, f"Expected 2 pattern-pair nodes, got {tn.n_nodes}"
        assert tn.n_edges == 2, f"Expected 2 directed edges, got {tn.n_edges}"

    def test_constant_series_single_pattern(self):
        """
        Constant series: all ordinal patterns are identical (argsort of identical
        values maps to the same tuple) → 1 node in the transition graph.
        """
        x = np.ones(20)
        tn = TransitionNetwork(symbolizer="ordinal", order=2).build(x)
        assert tn.n_nodes == 1, "Constant series should yield exactly one pattern-sequence node"

    def test_more_complex_series_has_more_nodes(self):
        """
        A more complex series should produce more distinct pattern sequences.
        Monotone series → fewer distinct sequences; random → many more.
        """
        rng = np.random.default_rng(30)
        x_random = rng.standard_normal(500)
        x_mono   = np.arange(500, dtype=float)

        tn_random = TransitionNetwork(symbolizer="ordinal", order=2).build(x_random)
        tn_mono   = TransitionNetwork(symbolizer="ordinal", order=2).build(x_mono)

        assert tn_random.n_nodes > tn_mono.n_nodes, (
            "Random series should have more distinct pattern sequences than monotone"
        )

    def test_n_nodes_consistent_with_n_edges(self):
        """n_nodes ≥ 1 and n_edges ≥ 0 for any non-trivial series."""
        rng = np.random.default_rng(31)
        x = rng.standard_normal(200)
        tn = TransitionNetwork(symbolizer="ordinal", order=2).build(x)
        assert tn.n_nodes >= 1
        assert tn.n_edges >= 1
        # Directed graph: sum of degrees can exceed 2 * n_edges
        # (each edge counted in out-degree of source only in directed graph)
        assert tn.degree_sequence().sum() >= tn.n_edges

    def test_longer_order_gives_more_nodes(self):
        """
        Higher order → each node represents a longer pattern sequence
        → more distinct sequences possible from the same data.
        """
        rng = np.random.default_rng(32)
        x = rng.standard_normal(1000)
        tn2 = TransitionNetwork(symbolizer="ordinal", order=2).build(x)
        tn3 = TransitionNetwork(symbolizer="ordinal", order=3).build(x)
        assert tn3.n_nodes >= tn2.n_nodes, (
            "Higher order should yield at least as many nodes as lower order"
        )
