"""
Property-based tests using Hypothesis.

These tests verify mathematical invariants that must hold for *any* valid
time series input, not just the hand-picked examples in test_graph_correctness.
Hypothesis generates hundreds of edge cases automatically:
  - n=2, n=3 (minimal series)
  - constant series (all values equal)
  - monotone increasing/decreasing
  - series with repeated values (ties)
  - series with extreme magnitudes
  - integer-valued series

Each test is a statement of the form "for all series x, property P holds."
If Hypothesis finds a counterexample it will shrink it to the smallest
failing case and report it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import Phase
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
from ts2net.distances.core import tsdist_dtw, tsdist_nmi, tsdist_voi


# ── Hypothesis strategies ──────────────────────────────────────────────────────

# A 1-D float64 array with finite values, length 2 to 200
finite_series = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=2, max_value=200),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# Same but longer (for statistical tests)
long_finite_series = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=50, max_value=300),
    elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)

# Two series of equal length (for pairwise distances)
@st.composite
def two_series(draw, min_len=10, max_len=100):
    n = draw(st.integers(min_value=min_len, max_value=max_len))
    x = draw(arrays(dtype=np.float64, shape=n,
                    elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)))
    y = draw(arrays(dtype=np.float64, shape=n,
                    elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)))
    return x, y


# ── HVG properties ────────────────────────────────────────────────────────────

class TestHVGProperties:

    @given(finite_series)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_n_nodes_equals_length(self, x):
        """HVG always has exactly n nodes."""
        assert HVG().build(x).n_nodes == len(x)

    @given(finite_series)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_every_node_degree_at_least_one(self, x):
        """Every node is connected to at least one other (no isolated nodes)."""
        d = HVG().build(x).degree_sequence()
        assert np.all(d >= 1), f"n={len(x)}: found node with degree 0"

    @given(finite_series)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_adjacent_always_connected(self, x):
        """Consecutive nodes are always directly visible (adjacent edge is always present)."""
        hvg = HVG().build(x)
        rows, cols, _ = hvg.edges_coo()
        edge_set = {(min(i, j), max(i, j)) for i, j in zip(rows.tolist(), cols.tolist())}
        for i in range(len(x) - 1):
            assert (i, i + 1) in edge_set, (
                f"Adjacent edge ({i},{i+1}) missing for n={len(x)}"
            )

    @given(finite_series)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_degree_sum_equals_twice_edges(self, x):
        """sum(degrees) == 2 * n_edges for undirected graph."""
        hvg = HVG().build(x)
        assert hvg.degree_sequence().sum() == 2 * hvg.n_edges

    @given(finite_series, st.floats(min_value=-100, max_value=100,
                                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_invariant_to_constant_shift(self, x, c):
        """
        HVG is invariant to adding a constant in exact arithmetic.

        In float64, adding c to values near zero can collapse tiny differences
        to zero (e.g. x[i]=3.56e-170, c=1.0 → x[i]+c rounds to exactly 1.0),
        changing structural ties and therefore the graph.  We skip series where
        any pairwise difference would be erased by the shift.
        """
        # Skip if any two values are so close that |c| would erase the difference
        diffs = np.diff(np.sort(x))
        min_gap = float(np.min(np.abs(diffs[diffs != 0]))) if np.any(diffs != 0) else 0.0
        assume(min_gap == 0.0 or abs(c) < min_gap * 1e10)

        r1, c1, _ = HVG().build(x).edges_coo()
        r2, c2, _ = HVG().build(x + c).edges_coo()
        edges1 = {(min(i, j), max(i, j)) for i, j in zip(r1.tolist(), c1.tolist())}
        edges2 = {(min(i, j), max(i, j)) for i, j in zip(r2.tolist(), c2.tolist())}
        assert edges1 == edges2

    @given(finite_series, st.floats(min_value=1e-3, max_value=1e3,
                                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_invariant_to_positive_scaling(self, x, s):
        """HVG is invariant to multiplying by a positive constant."""
        r1, c1, _ = HVG().build(x).edges_coo()
        r2, c2, _ = HVG().build(x * s).edges_coo()
        edges1 = {(min(i,j), max(i,j)) for i,j in zip(r1.tolist(), c1.tolist())}
        edges2 = {(min(i,j), max(i,j)) for i,j in zip(r2.tolist(), c2.tolist())}
        assert edges1 == edges2

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_adjacency_matrix_symmetric(self, x):
        """Adjacency matrix is symmetric for undirected HVG."""
        A = HVG().build(x).adjacency_matrix(format="dense")
        assert np.allclose(A, A.T)

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_adjacency_diagonal_zero(self, x):
        """No self-loops: diagonal must be zero."""
        A = HVG().build(x).adjacency_matrix(format="dense")
        assert np.allclose(np.diag(A), 0.0)

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_degrees_output_matches_edges_output(self, x):
        """output='degrees' produces same degree sequence as output='edges'."""
        d_edges   = HVG(output="edges").build(x).degree_sequence()
        d_degrees = HVG(output="degrees").build(x).degree_sequence()
        assert np.array_equal(d_edges, d_degrees)

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_hvg_subset_of_nvg(self, x):
        """Every HVG edge is also an NVG edge (HVG ⊆ NVG)."""
        r1, c1, _ = HVG().build(x).edges_coo()
        r2, c2, _ = NVG().build(x).edges_coo()
        hvg_edges = {(min(i,j), max(i,j)) for i,j in zip(r1.tolist(), c1.tolist())}
        nvg_edges = {(min(i,j), max(i,j)) for i,j in zip(r2.tolist(), c2.tolist())}
        assert hvg_edges.issubset(nvg_edges), (
            f"n={len(x)}: HVG has edges not in NVG: {hvg_edges - nvg_edges}"
        )


# ── NVG properties ────────────────────────────────────────────────────────────

class TestNVGProperties:

    @given(finite_series)
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_n_nodes_equals_length(self, x):
        assert NVG().build(x).n_nodes == len(x)

    @given(finite_series)
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_every_node_degree_at_least_one(self, x):
        d = NVG().build(x).degree_sequence()
        assert np.all(d >= 1)

    @given(finite_series)
    @settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_degree_sum_equals_twice_edges(self, x):
        nvg = NVG().build(x)
        assert nvg.degree_sequence().sum() == 2 * nvg.n_edges

    @given(long_finite_series, st.integers(min_value=2, max_value=50))
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_limit_respected(self, x, limit):
        """No edge in a limited NVG spans more than `limit` steps."""
        nvg = NVG(limit=limit).build(x)
        rows, cols, _ = nvg.edges_coo()
        if len(rows) > 0:
            gaps = np.abs(cols - rows)
            assert gaps.max() <= limit, (
                f"limit={limit} violated: max gap = {gaps.max()}"
            )


# ── RecurrenceNetwork properties ──────────────────────────────────────────────

class TestRecurrenceNetworkProperties:

    @given(finite_series)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_n_nodes_equals_length(self, x):
        rn = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x)
        assert rn.n_nodes == len(x)

    @given(finite_series)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_adjacency_symmetric(self, x):
        """Recurrence networks are symmetric (undirected)."""
        A = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x).adjacency_matrix(format="dense")
        assert np.allclose(A, A.T)

    @given(finite_series)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_large_epsilon_gives_complete_or_near_complete(self, x):
        """epsilon >> max_pairwise_distance → complete graph."""
        span = float(np.max(x) - np.min(x)) + 1.0
        rn = RecurrenceNetwork(rule="epsilon", epsilon=span * 2).build(x)
        n = len(x)
        # Complete undirected graph has n*(n-1)/2 edges
        assert rn.n_edges == n * (n - 1) // 2, (
            f"n={n}, epsilon={span*2}: expected {n*(n-1)//2} edges, got {rn.n_edges}"
        )

    @given(finite_series)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_degree_sum_equals_twice_edges(self, x):
        rn = RecurrenceNetwork(rule="epsilon", epsilon=0.5).build(x)
        assert rn.degree_sequence().sum() == 2 * rn.n_edges


# ── Weighted graph properties ─────────────────────────────────────────────────

class TestWeightedGraphProperties:

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_absdiff_weights_all_finite_nonneg(self, x):
        """absdiff weights are all finite and ≥ 0."""
        _, _, w = HVG(weighted="absdiff").build(x).edges_coo()
        assert w is not None
        assert np.all(np.isfinite(w)), "absdiff weights contain NaN or Inf"
        assert np.all(w >= 0.0), "absdiff weights must be non-negative"

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_time_gap_weights_equal_j_minus_i(self, x):
        """time_gap weight for edge (i,j) equals |j - i|."""
        hvg = HVG(weighted="time_gap").build(x)
        rows, cols, weights = hvg.edges_coo()
        expected = np.abs(cols - rows).astype(float)
        assert np.allclose(weights, expected), (
            "time_gap weight ≠ |j - i| for some edges"
        )

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_absdiff_weight_matches_formula(self, x):
        """absdiff weight for edge (i,j) equals |x[i] - x[j]|."""
        hvg = HVG(weighted="absdiff").build(x)
        rows, cols, weights = hvg.edges_coo()
        expected = np.abs(x[rows] - x[cols])
        assert np.allclose(weights, expected, rtol=1e-5), (
            "absdiff weight ≠ |x[i] - x[j]| for some edges"
        )

    @given(finite_series)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_weighted_adjacency_nonneg_and_finite(self, x):
        """All entries in a weighted adjacency matrix are finite and ≥ 0."""
        A = HVG(weighted="absdiff").build(x).adjacency_matrix(format="dense")
        assert np.all(np.isfinite(A))
        assert np.all(A >= 0.0)


# ── Distance function properties ──────────────────────────────────────────────

class TestDistanceProperties:

    @given(two_series(min_len=20, max_len=80))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_dtw_self_distance_zero(self, xy):
        x, _ = xy
        X = np.vstack([x, x])
        D = tsdist_dtw(X)
        assert np.isclose(D[0, 1], 0.0, atol=1e-10), f"DTW(x,x) = {D[0,1]} ≠ 0"

    @given(two_series(min_len=20, max_len=80))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_dtw_symmetric(self, xy):
        x, y = xy
        X = np.vstack([x, y])
        D = tsdist_dtw(X)
        assert np.isclose(D[0, 1], D[1, 0], rtol=1e-6), (
            f"DTW not symmetric: D[0,1]={D[0,1]} ≠ D[1,0]={D[1,0]}"
        )

    @given(two_series(min_len=20, max_len=80))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_dtw_non_negative(self, xy):
        x, y = xy
        X = np.vstack([x, y])
        D = tsdist_dtw(X)
        assert np.all(D >= 0.0), "DTW distance is negative"

    @given(two_series(min_len=30, max_len=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_voi_non_negative(self, xy):
        x, y = xy
        # Skip degenerate cases (constant series → zero entropy → 0/0 edge cases)
        assume(np.std(x) > 1e-6 and np.std(y) > 1e-6)
        v = tsdist_voi(x, y)
        assert v >= 0.0, f"VOI = {v} < 0"

    @given(two_series(min_len=30, max_len=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_voi_symmetric(self, xy):
        x, y = xy
        assume(np.std(x) > 1e-6 and np.std(y) > 1e-6)
        assert np.isclose(tsdist_voi(x, y), tsdist_voi(y, x), atol=1e-10)

    @given(long_finite_series)
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_nmi_self_distance_zero(self, x):
        """NMI(x, x) == 0 (identical series → zero distance)."""
        assume(np.std(x) > 1e-6)
        assert np.isclose(tsdist_nmi(x, x), 0.0, atol=1e-10)
