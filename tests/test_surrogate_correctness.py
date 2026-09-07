"""Correctness tests for the IAAFT surrogate and its legacy predecessor.

These replace the ``TestIaaftDefect`` characterization tests. IAAFT is defined
by two properties, and both are asserted here for the Rust path and the NumPy
fallback:

1. the surrogate is an exact permutation of the input, so the value
   distribution is preserved exactly; and
2. its power spectrum approaches the input's.

Bit-for-bit parity between Rust and Python is not achievable and is not
asserted: the two draw their initial random permutation from different
generators (``StdRng`` seeded with a u64 versus ``numpy.random.Generator``).
The tests therefore assert that both satisfy the same invariants within
explicit tolerances.
"""

import numpy as np
import pytest

import ts2net  # noqa: F401  -- aliases the compiled extension as `ts2net_rs`
import ts2net.stats.stats as st
import ts2net_rs

# Largest relative L2 error between the surrogate's amplitude spectrum and the
# original's that we accept. IAAFT converges to a fixed point trading exact
# amplitudes against an exact spectrum, so this is small but never zero;
# measured values are ~0.05 (Rust) and ~0.06 (NumPy) for the fixture below.
SPECTRAL_TOL = 0.15

# The surrogate must beat a shuffle of the same values by this factor. A
# shuffle has an identical value distribution and a destroyed spectrum, which
# makes it the right control for "did the spectral step do anything".
SHUFFLE_MARGIN = 5.0


def _signal(n=200):
    return np.sin(np.arange(n) * 0.3)


def _spectrum(x):
    return np.abs(np.fft.rfft(x))


def _spectral_relerr(x, surrogate):
    a, b = _spectrum(x), _spectrum(surrogate)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


@pytest.fixture
def numpy_fallback():
    """Force ts2net.stats.stats onto its NumPy path for the duration of a test."""
    saved = (st._iaaft_rs, st._iaaft_legacy_rs)
    st._iaaft_rs = None
    st._iaaft_legacy_rs = None
    try:
        yield st
    finally:
        st._iaaft_rs, st._iaaft_legacy_rs = saved


class TestIaaftPreservesTheValueDistribution:
    def test_rust_surrogate_is_a_permutation_of_the_input(self):
        x = _signal()
        surrogate = ts2net_rs.iaaft(x, 50, 3)
        np.testing.assert_allclose(np.sort(x), np.sort(surrogate), atol=1e-12)

    def test_numpy_surrogate_is_a_permutation_of_the_input(self, numpy_fallback):
        x = _signal()
        surrogate = numpy_fallback.iaaft(x, iters=50, rng=3)
        np.testing.assert_allclose(np.sort(x), np.sort(surrogate), atol=1e-12)

    def test_a_repeated_value_keeps_its_multiplicity(self):
        # Ties exercise the rank-matching step; multiplicities must survive.
        x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
        surrogate = ts2net_rs.iaaft(x, 30, 7)
        np.testing.assert_allclose(np.sort(x), np.sort(surrogate), atol=1e-12)


class TestIaaftTargetsTheOriginalSpectrum:
    def test_rust_beats_a_shuffle_of_the_same_values(self):
        x = _signal()
        shuffled = np.random.default_rng(0).permutation(x)
        got = _spectral_relerr(x, ts2net_rs.iaaft(x, 200, 5))
        baseline = _spectral_relerr(x, shuffled)
        assert got < SPECTRAL_TOL
        assert got < baseline / SHUFFLE_MARGIN

    def test_numpy_beats_a_shuffle_of_the_same_values(self, numpy_fallback):
        x = _signal()
        shuffled = np.random.default_rng(0).permutation(x)
        got = _spectral_relerr(x, numpy_fallback.iaaft(x, iters=200, rng=5))
        baseline = _spectral_relerr(x, shuffled)
        assert got < SPECTRAL_TOL
        assert got < baseline / SHUFFLE_MARGIN


class TestCrossLanguageAgreement:
    """Both implementations satisfy the same contract within explicit tolerances."""

    def test_both_preserve_the_distribution_and_target_the_spectrum(
        self, numpy_fallback
    ):
        x = _signal()
        rust = np.asarray(ts2net_rs.iaaft(x, 200, 5), dtype=float)
        python = np.asarray(numpy_fallback.iaaft(x, iters=200, rng=5), dtype=float)

        # Property 1, exactly, in both.
        np.testing.assert_allclose(np.sort(x), np.sort(rust), atol=1e-12)
        np.testing.assert_allclose(np.sort(x), np.sort(python), atol=1e-12)

        # Property 2, to the same explicit tolerance, in both.
        assert _spectral_relerr(x, rust) < SPECTRAL_TOL
        assert _spectral_relerr(x, python) < SPECTRAL_TOL

    def test_neither_implementation_returns_the_input_unchanged(self, numpy_fallback):
        x = _signal()
        assert not np.allclose(x, ts2net_rs.iaaft(x, 50, 3))
        assert not np.allclose(x, numpy_fallback.iaaft(x, iters=50, rng=3))


class TestDeterminism:
    def test_rust_is_reproducible_for_a_seed(self):
        x = _signal()
        np.testing.assert_array_equal(
            ts2net_rs.iaaft(x, 25, 11), ts2net_rs.iaaft(x, 25, 11)
        )

    def test_different_seeds_give_different_surrogates(self):
        x = _signal()
        assert not np.array_equal(
            ts2net_rs.iaaft(x, 25, 11), ts2net_rs.iaaft(x, 25, 12)
        )


class TestIaaftLegacy:
    """The pre-0.10 behaviour, retained only for reproducing published results."""

    # Captured from a build of the pre-refactor extension at commit d24cec9,
    # for np.sin(np.arange(200) * 0.3) with iters=20, seed=3.
    PRE_FIX_HEAD = [0.041038, 0.276636, 0.531711]

    def test_rust_legacy_reproduces_the_pre_fix_values_exactly(self):
        x = _signal()
        got = ts2net_rs.iaaft_legacy(x, 20, 3)
        np.testing.assert_array_equal(np.round(got[:3], 6), self.PRE_FIX_HEAD)

    def test_legacy_keeps_the_rank_ordering_and_not_the_amplitudes(self):
        x = _signal()
        got = ts2net_rs.iaaft_legacy(x, 50, 3)
        np.testing.assert_array_equal(
            np.argsort(np.argsort(x)), np.argsort(np.argsort(got))
        )
        assert not np.allclose(np.sort(x), np.sort(got))

    def test_numpy_legacy_has_the_same_semantics(self, numpy_fallback):
        x = _signal()
        got = numpy_fallback.iaaft_legacy(x, iters=50, rng=3)
        np.testing.assert_array_equal(
            np.argsort(np.argsort(x)), np.argsort(np.argsort(got))
        )
        assert not np.allclose(np.sort(x), np.sort(got))

    def test_legacy_is_exported_from_the_stats_package(self):
        from ts2net.stats import iaaft_legacy

        assert callable(iaaft_legacy)


class TestGenerateSurrogateUsesTheFixedAlgorithm:
    def test_iaaft_method_now_preserves_the_distribution(self):
        from ts2net.stats.null_models import generate_surrogate

        x = _signal(128)
        surrogate = generate_surrogate(
            x, method="iaaft", iters=50, rng=np.random.default_rng(42)
        )
        np.testing.assert_allclose(np.sort(x), np.sort(surrogate), atol=1e-12)
