"""
API-consistency tests — verify that different entry points and output modes
agree with each other and with hand-computable ground truth.

Covers gaps not addressed by existing tests:
  - Weight mode exact values (absdiff, time_gap, slope)
  - build() == fit_transform() edge agreement
  - output='degrees' degrees match output='edges' degrees
  - adjacency_matrix sparse vs dense agreement
  - NVG limit parameter: no edge exceeds the horizon
  - stats() avg_degree == degree_sequence().mean()
  - Distance matrix symmetry for tsdist_cor and tsdist_ccf
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ts2net import HVG, NVG, RecurrenceNetwork


# ── helpers ────────────────────────────────────────────────────────────────────

def edge_dict(g) -> dict[tuple[int, int], float | None]:
    """Return {(min_i, max_j): weight} for the graph's edges."""
    rows, cols, weights = g.edges_coo()
    if weights is None:
        return {(min(i, j), max(i, j)): None
                for i, j in zip(rows.tolist(), cols.tolist())}
    return {(min(i, j), max(i, j)): float(w)
            for i, j, w in zip(rows.tolist(), cols.tolist(), weights.tolist())}


# ══════════════════════════════════════════════════════════════════════════════
# Weight mode exact values
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightModeValues:
    """Verify weight mode formulas against hand-computed expected values."""

    X = np.array([1.0, 3.0, 2.0, 4.0, 1.0])

    # Hand-derived HVG edges for X = [1, 3, 2, 4, 1]
    # (0,1): adjacent                  absdiff = |3-1| = 2   time_gap = 1  slope = (3-1)/1 = 2
    # (1,2): adjacent                  absdiff = |2-3| = 1   time_gap = 1  slope = (2-3)/1 = -1
    # (1,3): x[2]=2 < min(3,4)=3 → ok  absdiff = |4-3| = 1   time_gap = 2  slope = (4-3)/2 = 0.5
    # (2,3): adjacent                  absdiff = |4-2| = 2   time_gap = 1  slope = (4-2)/1 = 2
    # (3,4): adjacent                  absdiff = |1-4| = 3   time_gap = 1  slope = (1-4)/1 = -3
    EXPECTED_ABSDIFF = {
        (0, 1): 2.0,
        (1, 2): 1.0,
        (1, 3): 1.0,
        (2, 3): 2.0,
        (3, 4): 3.0,
    }
    EXPECTED_TIME_GAP = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (1, 3): 2.0,
        (2, 3): 1.0,
        (3, 4): 1.0,
    }
    EXPECTED_SLOPE = {
        (0, 1):  2.0,
        (1, 2): -1.0,
        (1, 3):  0.5,
        (2, 3):  2.0,
        (3, 4): -3.0,
    }

    def test_absdiff_values(self):
        """w(i,j) == |x[i] - x[j]| for every edge."""
        hvg = HVG(weighted="absdiff").build(self.X)
        actual = edge_dict(hvg)
        assert set(actual.keys()) == set(self.EXPECTED_ABSDIFF.keys()), (
            f"Edge sets differ.\nExpected: {set(self.EXPECTED_ABSDIFF)}\nGot: {set(actual)}"
        )
        for (i, j), expected_w in self.EXPECTED_ABSDIFF.items():
            assert_allclose(actual[(i, j)], expected_w, rtol=1e-6,
                            err_msg=f"absdiff weight wrong for edge ({i},{j})")

    def test_time_gap_values(self):
        """w(i,j) == j - i (temporal distance) for every edge."""
        hvg = HVG(weighted="time_gap").build(self.X)
        actual = edge_dict(hvg)
        for (i, j), expected_w in self.EXPECTED_TIME_GAP.items():
            assert_allclose(actual[(i, j)], expected_w, rtol=1e-6,
                            err_msg=f"time_gap weight wrong for edge ({i},{j})")

    def test_slope_values(self):
        """w(i,j) == (x[j] - x[i]) / (j - i) for every edge."""
        hvg = HVG(weighted="slope").build(self.X)
        actual = edge_dict(hvg)
        for (i, j), expected_w in self.EXPECTED_SLOPE.items():
            assert_allclose(actual[(i, j)], expected_w, rtol=1e-6,
                            err_msg=f"slope weight wrong for edge ({i},{j})")

    def test_absdiff_always_non_negative(self):
        """absdiff weights are always ≥ 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        _, _, weights = HVG(weighted="absdiff").build(x).edges_coo()
        assert weights is not None
        assert np.all(weights >= 0.0)

    def test_time_gap_equals_j_minus_i(self):
        """time_gap weight equals j - i for every edge."""
        rng = np.random.default_rng(43)
        x = rng.standard_normal(50)
        hvg = HVG(weighted="time_gap").build(x)
        rows, cols, weights = hvg.edges_coo()
        expected_gaps = np.abs(cols - rows).astype(float)
        assert_allclose(weights, expected_gaps, rtol=1e-6)

    def test_unweighted_edges_have_no_weights(self):
        """Unweighted graph: edges_coo() returns None weights."""
        x = np.random.default_rng(44).standard_normal(30)
        _, _, weights = HVG().build(x).edges_coo()
        assert weights is None, "Unweighted HVG should return None weights"


# ══════════════════════════════════════════════════════════════════════════════
# build() vs fit_transform() agreement
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildVsFitTransform:
    """build() and fit_transform() must produce the same edges."""

    @pytest.mark.parametrize("cls,kwargs", [
        (HVG, {}),
        (HVG, {"weighted": "absdiff"}),
        (HVG, {"directed": True}),
        (NVG, {}),
        (NVG, {"limit": 20}),
    ])
    def test_same_edges(self, cls, kwargs):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(80)
        via_build = cls(**kwargs).build(x)
        via_ft    = cls(**kwargs)
        G_nx = via_ft.fit_transform(x)

        rows_b, cols_b, _ = via_build.edges_coo()
        edges_build = {(min(i, j), max(i, j)) for i, j in zip(rows_b.tolist(), cols_b.tolist())}
        edges_ft    = {(min(u, v), max(u, v)) for u, v in G_nx.edges()}

        assert edges_build == edges_ft, (
            f"{cls.__name__}({kwargs}): build() and fit_transform() produced different edges.\n"
            f"Only in build: {edges_build - edges_ft}\n"
            f"Only in fit_transform: {edges_ft - edges_build}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Output mode consistency
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputModeConsistency:
    """output='degrees' must produce the same degree sequence as output='edges'."""

    @pytest.mark.parametrize("cls", [HVG, NVG])
    def test_degrees_output_matches_edges_output(self, cls):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(150)
        kwargs = {"limit": 50} if cls is NVG else {}

        g_edges   = cls(**kwargs, output="edges").build(x)
        g_degrees = cls(**kwargs, output="degrees").build(x)

        assert_array_equal(
            g_edges.degree_sequence(),
            g_degrees.degree_sequence(),
            err_msg=(
                f"{cls.__name__}: output='degrees' and output='edges' produced "
                "different degree sequences"
            ),
        )

    @pytest.mark.parametrize("cls", [HVG, NVG])
    def test_stats_output_avg_degree_matches(self, cls):
        """stats output avg_degree should equal edges output avg_degree."""
        rng = np.random.default_rng(6)
        x = rng.standard_normal(100)
        kwargs = {"limit": 50} if cls is NVG else {}

        g_edges = cls(**kwargs, output="edges").build(x)
        g_stats = cls(**kwargs, output="stats").build(x)

        expected_avg = g_edges.degree_sequence().mean()
        actual_avg   = g_stats.stats().get("avg_degree") or g_stats.stats().get("mean_degree")

        if actual_avg is not None:
            assert_allclose(actual_avg, expected_avg, rtol=1e-4,
                            err_msg=f"{cls.__name__}: stats avg_degree mismatch")


# ══════════════════════════════════════════════════════════════════════════════
# Adjacency matrix sparse / dense agreement
# ══════════════════════════════════════════════════════════════════════════════

class TestAdjacencyMatrixFormats:
    """sparse and dense adjacency matrix must contain the same values."""

    @pytest.mark.parametrize("cls,kwargs", [
        (HVG, {}),
        (HVG, {"weighted": "absdiff"}),
        (NVG, {"limit": 30}),
    ])
    def test_sparse_dense_equal(self, cls, kwargs):
        rng = np.random.default_rng(10)
        x = rng.standard_normal(40)
        g = cls(**kwargs).build(x)
        A_sparse = g.adjacency_matrix(format="sparse")
        A_dense  = g.adjacency_matrix(format="dense")
        assert_allclose(A_sparse.toarray(), A_dense, rtol=1e-6,
                        err_msg=f"{cls.__name__}: sparse.toarray() ≠ dense")

    def test_dense_is_symmetric_for_undirected(self):
        x = np.random.default_rng(11).standard_normal(30)
        A = HVG().build(x).adjacency_matrix(format="dense")
        assert_allclose(A, A.T, rtol=1e-6, err_msg="HVG adjacency must be symmetric")

    def test_weighted_adjacency_contains_correct_values(self):
        """A[i,j] == absdiff weight for edge (i,j)."""
        x = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        hvg = HVG(weighted="absdiff").build(x)
        A = hvg.adjacency_matrix(format="dense")

        rows, cols, weights = hvg.edges_coo()
        for i, j, w in zip(rows.tolist(), cols.tolist(), weights.tolist()):
            assert_allclose(A[i, j], w, rtol=1e-6,
                            err_msg=f"Dense A[{i},{j}] ≠ edge weight {w}")
            assert_allclose(A[j, i], w, rtol=1e-6,
                            err_msg=f"Dense A[{j},{i}] ≠ edge weight {w} (symmetry)")

    def test_adjacency_diagonal_zero(self):
        """No self-loops: diagonal of adjacency matrix must be zero."""
        x = np.random.default_rng(12).standard_normal(20)
        for cls in (HVG, NVG):
            A = cls().build(x).adjacency_matrix(format="dense")
            assert_allclose(np.diag(A), 0.0, atol=1e-10,
                            err_msg=f"{cls.__name__}: diagonal must be zero (no self-loops)")


# ══════════════════════════════════════════════════════════════════════════════
# NVG limit parameter
# ══════════════════════════════════════════════════════════════════════════════

class TestNVGLimit:
    """With limit=k, no edge should span more than k time steps."""

    @pytest.mark.parametrize("limit", [5, 10, 20])
    def test_no_edge_exceeds_limit(self, limit):
        rng = np.random.default_rng(20)
        x = rng.standard_normal(100)
        nvg = NVG(limit=limit).build(x)
        rows, cols, _ = nvg.edges_coo()
        gaps = np.abs(cols - rows)
        violations = gaps[gaps > limit]
        assert len(violations) == 0, (
            f"NVG(limit={limit}): found {len(violations)} edges exceeding the limit. "
            f"Max gap = {gaps.max()}"
        )

    def test_limit_reduces_edge_count(self):
        """Fewer edges with a tight limit than without."""
        rng = np.random.default_rng(21)
        x = rng.standard_normal(200)
        nvg_full    = NVG().build(x)
        nvg_limited = NVG(limit=10).build(x)
        assert nvg_limited.n_edges < nvg_full.n_edges, (
            "NVG with limit=10 should have fewer edges than unconstrained NVG"
        )


# ══════════════════════════════════════════════════════════════════════════════
# stats() values agree with degree_sequence()
# ══════════════════════════════════════════════════════════════════════════════

class TestStatsConsistency:
    """stats() reported values must agree with what the graph actually contains."""

    @pytest.mark.parametrize("cls,kwargs", [
        (HVG, {}),
        (NVG, {"limit": 50}),
    ])
    def test_avg_degree_matches(self, cls, kwargs):
        rng = np.random.default_rng(30)
        x = rng.standard_normal(100)
        g = cls(**kwargs).build(x)
        d = g.degree_sequence()
        s = g.stats()

        # Accept either key name
        avg_from_stats = s.get("avg_degree") or s.get("mean_degree") or s.get("average_degree")
        if avg_from_stats is None:
            pytest.skip(f"stats() does not return an avg_degree key for {cls.__name__}")

        assert_allclose(avg_from_stats, d.mean(), rtol=1e-4,
                        err_msg=f"{cls.__name__}: stats avg_degree ≠ degree_sequence().mean()")

    @pytest.mark.parametrize("cls,kwargs", [
        (HVG, {}),
        (NVG, {"limit": 50}),
    ])
    def test_n_nodes_matches(self, cls, kwargs):
        rng = np.random.default_rng(31)
        x = rng.standard_normal(100)
        g = cls(**kwargs).build(x)
        s = g.stats()
        n_from_stats = s.get("n_nodes") or s.get("nodes")
        if n_from_stats is None:
            pytest.skip("stats() does not return n_nodes")
        assert n_from_stats == g.n_nodes

    @pytest.mark.parametrize("cls,kwargs", [
        (HVG, {}),
        (NVG, {"limit": 50}),
    ])
    def test_n_edges_matches(self, cls, kwargs):
        rng = np.random.default_rng(32)
        x = rng.standard_normal(100)
        g = cls(**kwargs).build(x)
        s = g.stats()
        n_from_stats = s.get("n_edges") or s.get("edges")
        if n_from_stats is None:
            pytest.skip("stats() does not return n_edges")
        assert n_from_stats == g.n_edges


# ══════════════════════════════════════════════════════════════════════════════
# Distance matrix symmetry
# ══════════════════════════════════════════════════════════════════════════════

class TestDistanceMatrixSymmetry:
    """All pairwise distance functions must produce symmetric matrices."""

    def _random_series(self, n=5, seed=0):
        return np.random.default_rng(seed).standard_normal((n, 80))

    def test_tsdist_cor_symmetric(self):
        from ts2net.distances.core import tsdist_cor
        X = self._random_series()
        D = tsdist_cor(X)
        assert_allclose(D, D.T, atol=1e-10)

    def test_tsdist_ccf_symmetric(self):
        from ts2net.distances.core import tsdist_ccf
        X = self._random_series()
        D = tsdist_ccf(X, max_lag=5)
        assert_allclose(D, D.T, atol=1e-10)

    def test_tsdist_dtw_symmetric(self):
        from ts2net.distances.core import tsdist_dtw
        X = self._random_series()
        D = tsdist_dtw(X)
        assert_allclose(D, D.T, atol=1e-10)

    def test_tsdist_nmi_symmetric(self):
        from ts2net.distances.core import tsdist_nmi
        rng = np.random.default_rng(50)
        x, y = rng.standard_normal(200), rng.standard_normal(200)
        assert_allclose(tsdist_nmi(x, y), tsdist_nmi(y, x), atol=1e-10)

    def test_tsdist_cor_self_distance_zero(self):
        from ts2net.distances.core import tsdist_cor
        X = self._random_series()
        D = tsdist_cor(X)
        assert_allclose(np.diag(D), 0.0, atol=1e-6,
                        err_msg="tsdist_cor: self-distance must be 0")

    def test_tsdist_dtw_self_distance_zero(self):
        from ts2net.distances.core import tsdist_dtw
        X = self._random_series()
        D = tsdist_dtw(X)
        assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_tsdist_nmi_self_distance_zero(self):
        from ts2net.distances.core import tsdist_nmi
        rng = np.random.default_rng(51)
        x = rng.standard_normal(200)
        assert_allclose(tsdist_nmi(x, x), 0.0, atol=1e-10)

    def test_tsdist_voi_non_negative_all_pairs(self):
        from ts2net.distances.core import tsdist_voi
        rng = np.random.default_rng(52)
        for _ in range(10):
            x = rng.standard_normal(100)
            y = rng.standard_normal(100)
            assert tsdist_voi(x, y) >= 0.0
