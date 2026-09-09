"""Characterization of the state-space reconstruction machinery.

`ts2net` has the pieces of a delay-embedding / manifold layer -- embedding,
embedding-dimension selection, neighbour queries, recurrence networks -- but no
type that owns them. This pins what currently holds.

The two defects this file originally captured, in both NumPy fallbacks for
embedding-dimension selection, are now fixed; the tests below assert their
correctness instead.

See docs/audits/state_space_reconstruction_audit.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts2net  # noqa: F401
import ts2net_rs
from ts2net.core import embed as core_embed
from ts2net.stats.threshold_sensitivity import _delay_embed


def lorenz_x(n: int = 4000, dt: float = 0.01, transient: int = 1000) -> np.ndarray:
    """x-component of the Lorenz system, Euler integrated. Deterministic."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    out = np.empty((n, 3))
    state = np.array([1.0, 1.0, 1.0])
    for i in range(n):
        x, y, z = state
        state = state + dt * np.array(
            [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        )
        out[i] = state
    series = out[transient:, 0]
    return (series - series.mean()) / series.std()


def sine(n: int = 3000, step: float = 0.05) -> np.ndarray:
    return np.sin(np.arange(n) * step)


def white_noise(n: int = 3000) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n)


class TestDelayEmbeddingConvention:
    """One convention, several implementations, and they agree.

    Every delay embedding in the package builds ``E[:, i] = x[i*tau : i*tau + L]``
    with ``L = n - (m-1)*tau``, so a row is a state vector and column ``i`` is
    the series lagged by ``i*tau``. This pins that, because a manifold layer
    would fix the convention once and everything else must match it.
    """

    @pytest.mark.parametrize("m, tau", [(2, 1), (3, 1), (4, 5), (2, 10), (6, 3)])
    def test_implementations_agree(self, m, tau):
        x = sine(400)
        np.testing.assert_array_equal(core_embed(x, m, tau), _delay_embed(x, m, tau))

    @pytest.mark.parametrize("m, tau", [(2, 1), (3, 4), (5, 2)])
    def test_shape_and_column_meaning(self, m, tau):
        x = np.arange(100.0)
        points = core_embed(x, m, tau)
        assert points.shape == (len(x) - (m - 1) * tau, m)
        for column in range(m):
            offset = column * tau
            np.testing.assert_array_equal(
                points[:, column], x[offset : offset + points.shape[0]]
            )

    def test_a_row_is_one_state_vector(self):
        x = np.arange(20.0)
        points = core_embed(x, 3, 2)
        np.testing.assert_array_equal(points[0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(points[5], [5.0, 7.0, 9.0])

    def test_too_short_a_series_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            core_embed(np.arange(5.0), m=4, tau=3)


class TestEmbeddingDimensionSelectionIsScientificallyCorrect:
    """The Rust path, which is what actually runs, against known systems.

    These are the contracts a `reconstruct()` API would rest on, so they are
    asserted against systems whose embedding dimension is known analytically
    rather than against recorded outputs.
    """

    def test_a_sine_unfolds_at_dimension_two(self):
        fnn = np.asarray(ts2net_rs.false_nearest_neighbors(sine(), 6, 15, 10.0, 2.0))
        assert fnn[1] < 0.01, f"a closed curve should unfold at m=2, got {fnn[1]}"

    def test_lorenz_unfolds_at_dimension_three(self):
        fnn = np.asarray(
            ts2net_rs.false_nearest_neighbors(lorenz_x(), 6, 10, 10.0, 2.0)
        )
        assert fnn[0] > 0.5, "m=1 should have many false neighbours"
        assert fnn[2] < 0.01, f"Lorenz should unfold at m=3, got {fnn[2]}"

    def test_white_noise_never_unfolds(self):
        fnn = np.asarray(
            ts2net_rs.false_nearest_neighbors(white_noise(), 6, 1, 10.0, 2.0)
        )
        assert fnn[-1] > 0.05, (
            f"noise has no finite embedding dimension, got {fnn[-1]}"
        )

    def test_cao_e2_separates_deterministic_from_stochastic(self):
        """E2 stays near 1 for stochastic data and departs from it otherwise."""
        _, e2_noise = ts2net_rs.cao_e1_e2(white_noise(), 6, 1)
        _, e2_lorenz = ts2net_rs.cao_e1_e2(lorenz_x(), 6, 10)
        assert abs(np.mean(e2_noise) - 1.0) < 0.25, np.mean(e2_noise)
        assert abs(np.mean(e2_lorenz) - 1.0) > 0.5, np.mean(e2_lorenz)


class TestNeighbourQueryDimensionCeiling:
    """`ts2net_rs.knn`/`radius` are monomorphised up to six dimensions.

    This is an implementation limit, not a mathematical one, and it binds
    exactly where a manifold layer would want fast neighbour queries: real
    signals often reconstruct above six dimensions.
    """

    @pytest.mark.parametrize("dim", [1, 3, 6])
    def test_supported_dimensions_work(self, dim):
        points = core_embed(sine(1000), dim, 5)
        idx, dist = ts2net_rs.knn(points, 3)
        assert idx.shape == (points.shape[0], 3)

    @pytest.mark.parametrize("dim", [7, 10])
    def test_higher_dimensions_are_rejected(self, dim):
        points = core_embed(sine(1000), dim, 5)
        with pytest.raises(ValueError, match="dimension up to 6"):
            ts2net_rs.knn(points, 3)

    def test_dimension_selection_is_unaffected_by_the_ceiling(self):
        """fnn/cao use brute-force neighbours, so they run past six."""
        fnn = np.asarray(
            ts2net_rs.false_nearest_neighbors(sine(2000), 10, 5, 10.0, 2.0)
        )
        assert len(fnn) == 9

    def test_recurrence_adjacency_is_unaffected_by_the_ceiling(self):
        points = core_embed(sine(1000), 8, 5)[:300]
        adjacency = np.asarray(ts2net_rs.rn_adj_epsilon(points, 1.0, "euclidean", 0))
        assert adjacency.shape == (300, 300)


class TestNumpyFallbacksMatchRust:
    """The NumPy fallbacks, fixed.

    Both previously took neighbour indices computed on the m-dimensional
    embedding, which has ``n-(m-1)*tau`` rows, and used them to index the
    (m+1)-dimensional embedding, which has ``tau`` fewer. Both always failed.
    `false_nearest_neighbors` additionally collapsed a per-point criterion to a
    single scalar via `np.linalg.norm` without `axis=`.

    Both now embed at a common length of ``n - m*tau``, as the Rust
    implementation does, and apply the criterion per point.
    """

    @staticmethod
    def _numpy_only(monkeypatch):
        import ts2net.stats.stats as stats_module

        monkeypatch.setattr(stats_module, "_fnn_rs", None)
        monkeypatch.setattr(stats_module, "_cao_rs", None)
        return stats_module

    SIGNALS = {
        "sine": sine(800, 0.05),
        "noisy_sine": (
            sine(800, 0.21)
            + 0.05 * np.random.default_rng(0).standard_normal(800)
        ),
        "white_noise": white_noise(800),
    }

    @pytest.mark.parametrize("name", sorted(SIGNALS))
    @pytest.mark.parametrize("m_max, tau", [(6, 1), (5, 3), (4, 7)])
    def test_fnn_is_bit_identical_to_rust(self, monkeypatch, name, m_max, tau):
        series = self.SIGNALS[name]
        expected = np.asarray(
            ts2net_rs.false_nearest_neighbors(series, m_max, tau, 10.0, 2.0)
        )
        stats_module = self._numpy_only(monkeypatch)
        got = np.asarray(
            stats_module.false_nearest_neighbors(series, m_max=m_max, tau=tau)
        )
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("name", sorted(SIGNALS))
    @pytest.mark.parametrize("m_max, tau", [(6, 1), (5, 3)])
    def test_cao_matches_rust_to_accumulation_error(
        self, monkeypatch, name, m_max, tau
    ):
        """Not bit-identical: the two sum the same terms in a different order."""
        series = self.SIGNALS[name]
        want_e1, want_e2 = [
            np.asarray(v) for v in ts2net_rs.cao_e1_e2(series, m_max, tau)
        ]
        stats_module = self._numpy_only(monkeypatch)
        got_e1, got_e2 = [
            np.asarray(v)
            for v in stats_module.cao_e1_e2(series, m_max=m_max, tau=tau)
        ]
        np.testing.assert_allclose(got_e1, want_e1, rtol=1e-9)
        np.testing.assert_allclose(got_e2, want_e2, rtol=1e-9, atol=1e-15)

    def test_the_fallback_gives_the_right_answer_on_known_systems(self, monkeypatch):
        """Independently correct, not merely equal to the Rust path."""
        stats_module = self._numpy_only(monkeypatch)
        fnn_sine = np.asarray(
            stats_module.false_nearest_neighbors(sine(), m_max=6, tau=15)
        )
        assert fnn_sine[1] < 0.01, f"sine should unfold at m=2, got {fnn_sine[1]}"

        fnn_lorenz = np.asarray(
            stats_module.false_nearest_neighbors(lorenz_x(), m_max=6, tau=10)
        )
        assert fnn_lorenz[0] > 0.5
        assert fnn_lorenz[2] < 0.01, f"Lorenz should unfold at m=3, got {fnn_lorenz[2]}"

        fnn_noise = np.asarray(
            stats_module.false_nearest_neighbors(white_noise(), m_max=6, tau=1)
        )
        assert fnn_noise[-1] > 0.05, "noise has no finite embedding dimension"

    def test_the_fallback_criterion_is_per_point_not_scalar(self, monkeypatch):
        """The second FNN defect: a scalar criterion forces 0.0 or 1.0 for all m.

        A signal that unfolds partway must produce intermediate fractions.
        """
        stats_module = self._numpy_only(monkeypatch)
        fractions = np.asarray(
            stats_module.false_nearest_neighbors(white_noise(), m_max=6, tau=1)
        )
        assert np.any((fractions > 0.01) & (fractions < 0.99)), fractions
        assert len(set(np.round(fractions, 6))) > 1, "criterion collapsed"

    @pytest.mark.parametrize("m_max", [0, 1])
    def test_m_max_below_two_is_rejected_like_the_rust_path(self, monkeypatch, m_max):
        stats_module = self._numpy_only(monkeypatch)
        with pytest.raises(ValueError, match="m_max"):
            stats_module.false_nearest_neighbors(sine(400), m_max=m_max)
        with pytest.raises(ValueError, match="m_max"):
            stats_module.cao_e1_e2(sine(400), m_max=m_max)

    def test_an_embedding_longer_than_the_series_degrades_gracefully(
        self, monkeypatch
    ):
        stats_module = self._numpy_only(monkeypatch)
        short = sine(30, 0.3)
        fnn = np.asarray(
            stats_module.false_nearest_neighbors(short, m_max=5, tau=20)
        )
        np.testing.assert_array_equal(fnn[1:], np.ones(3))
        e1, e2 = stats_module.cao_e1_e2(short, m_max=5, tau=20)
        assert np.all(np.isnan(np.asarray(e1)[1:]))

    def test_the_rust_path_is_still_the_default(self):
        import ts2net.stats.stats as stats_module

        assert stats_module._fnn_rs is not None
        assert stats_module._cao_rs is not None
