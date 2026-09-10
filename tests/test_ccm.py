"""Contracts for convergent cross mapping.

CCM is easy to over-read, so these tests are organised around the five claims
`CouplingVerdict` separates rather than around a single skill number. The
fixture is the coupled logistic system from Sugihara et al. (2012), where the
coupling is one-way by construction and the correct answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

import ts2net
from ts2net.ccm import CONVERGENCE_TOLERANCE, ccm, ccm_test, cross_map_curve


def coupled_logistic(
    n: int = 1500,
    b_xy: float = 0.0,
    b_yx: float = 0.32,
    r_x: float = 3.8,
    r_y: float = 3.5,
    burn: int = 500,
):
    """Two logistic maps. ``b_yx`` is the strength of x's effect on y.

    With ``b_xy = 0`` the coupling is one-way: x drives y and y does not drive
    x. Deterministic given the initial condition.
    """
    x = np.empty(n)
    y = np.empty(n)
    x[0], y[0] = 0.4, 0.2
    for t in range(n - 1):
        x[t + 1] = np.clip(x[t] * (r_x - r_x * x[t] - b_xy * y[t]), 1e-6, 1 - 1e-6)
        y[t + 1] = np.clip(y[t] * (r_y - r_y * y[t] - b_yx * x[t]), 1e-6, 1 - 1e-6)
    return x[burn:], y[burn:]


def independent_pair(n: int = 1000, seed: int = 0):
    """Two unrelated but individually autocorrelated series.

    Autocorrelation is what makes a naive cross map produce skill from nothing,
    so this is the fixture that matters for the Theiler window.
    """
    rng = np.random.default_rng(seed)
    def ar1(rho=0.95):
        out = np.empty(n)
        out[0] = rng.standard_normal()
        for t in range(n - 1):
            out[t + 1] = rho * out[t] + rng.standard_normal()
        return out
    return ar1(), ar1()


class TestOneWayCouplingIsDetected:
    def test_the_driving_direction_has_higher_skill(self):
        x, y = coupled_logistic()
        verdict = ccm(x, y, dimension=3, delay=1, n_draws=6)
        assert verdict.forward.terminal_skill > verdict.backward.terminal_skill
        assert verdict.forward.terminal_skill > 0.8

    def test_the_result_is_asymmetric(self):
        x, y = coupled_logistic()
        verdict = ccm(x, y, dimension=3, delay=1, n_draws=6)
        assert verdict.claims()["asymmetric"]

    def test_skill_converges_with_library_size(self):
        x, y = coupled_logistic()
        result = cross_map_curve(x, y, dimension=3, delay=1, n_draws=6)
        assert result.converged
        assert result.convergence_delta > 0.1
        assert result.skill[-1] > result.skill[0]

    def test_ccm_finds_what_correlation_misses(self):
        """The reason the method exists."""
        x, y = coupled_logistic()
        verdict = ccm(x, y, dimension=3, delay=1, n_draws=6)
        assert abs(verdict.pearson_r) < 0.4, verdict.pearson_r
        assert verdict.forward.terminal_skill > 0.8
        assert verdict.forward.terminal_skill > abs(verdict.pearson_r) * 2

    def test_stronger_coupling_gives_more_skill(self):
        skills = []
        for strength in (0.02, 0.1, 0.32):
            x, y = coupled_logistic(b_yx=strength)
            skills.append(
                cross_map_curve(x, y, dimension=3, delay=1, n_draws=6).terminal_skill
            )
        assert skills == sorted(skills), skills


class TestUncoupledSeriesAreNotCalledCoupled:
    def test_independent_autocorrelated_series_show_little_skill(self):
        x, y = independent_pair()
        result = cross_map_curve(x, y, dimension=3, delay=1, n_draws=6)
        assert result.terminal_skill < 0.5, result.terminal_skill

    def test_the_theiler_window_is_what_suppresses_spurious_skill(self):
        """Neighbours in a smooth trajectory are neighbours in time.

        Without temporal exclusion the estimate is an interpolation of the
        target's own past, and autocorrelation alone manufactures skill.
        """
        x, y = independent_pair()
        without = cross_map_curve(
            x, y, dimension=3, delay=1, theiler=0, n_draws=6
        ).terminal_skill
        with_window = cross_map_curve(
            x, y, dimension=3, delay=1, theiler=30, n_draws=6
        ).terminal_skill
        assert with_window < without, (
            f"exclusion should reduce spurious skill: {without:.3f} -> "
            f"{with_window:.3f}"
        )

    def test_the_default_exclusion_is_not_zero(self):
        x, y = independent_pair()
        result = cross_map_curve(x, y, dimension=3, delay=4, n_draws=4)
        assert result.theiler == 4, "exclusion should default to the delay"


class TestTheFiveClaimsAreSeparate:
    def test_claims_are_reported_independently(self):
        x, y = coupled_logistic()
        claims = ccm(x, y, dimension=3, delay=1, n_draws=6).claims()
        assert set(claims) == {
            "correlated",
            "predictive",
            "convergent",
            "asymmetric",
            "significant",
        }

    def test_significance_is_none_when_untested_not_false(self):
        """An untested claim must not read as a negative result."""
        x, y = coupled_logistic()
        verdict = ccm(x, y, dimension=3, delay=1, n_draws=6)
        assert verdict.claims()["significant"] is None
        assert "not tested against a surrogate null" in verdict.summary()

    def test_a_non_converging_relationship_says_so(self):
        x, y = independent_pair()
        verdict = ccm(x, y, dimension=3, delay=1, n_draws=6)
        if not verdict.forward.converged:
            assert "does NOT converge" in verdict.summary()

    def test_convergence_is_measured_against_available_headroom(self):
        """A relationship starting high need not rise as far as one starting low."""
        assert 0.0 < CONVERGENCE_TOLERANCE < 1.0


class TestSurrogateSignificance:
    def test_one_way_coupling_survives_an_iaaft_null(self):
        x, y = coupled_logistic()
        verdict = ccm_test(
            x, y, n_surrogates=40, dimension=3, delay=1, n_draws=3, null="iaaft"
        )
        assert verdict.forward_p is not None
        assert verdict.claims()["significant"], verdict.summary()
        assert verdict.n_surrogates == 40
        assert verdict.null_model == "iaaft"

    def test_the_p_value_is_add_one_smoothed(self):
        x, y = coupled_logistic()
        verdict = ccm_test(
            x, y, n_surrogates=20, dimension=3, delay=1, n_draws=2, null="shuffle"
        )
        assert verdict.forward_p >= 1 / 21
        assert verdict.forward_p > 0.0

    def test_an_unknown_null_is_rejected(self):
        # n must exceed the 500-sample burn-in the fixture discards.
        x, y = coupled_logistic(n=900)
        with pytest.raises(ValueError, match="unknown null model"):
            ccm_test(x, y, n_surrogates=2, null="wishful", dimension=3, delay=1)

    def test_n_surrogates_must_be_positive(self):
        x, y = coupled_logistic(n=900)
        with pytest.raises(ValueError, match="at least 1"):
            ccm_test(x, y, n_surrogates=0, dimension=3, delay=1)


class TestParameterHandlingAndSurface:
    def test_parameters_default_to_a_reconstruction_of_the_effect(self):
        x, y = coupled_logistic()
        verdict = ccm(x, y, n_draws=3)
        assert verdict.forward.embedding_dimension >= 1
        assert verdict.forward.delay >= 1

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            ccm(np.zeros(100), np.zeros(90), dimension=2, delay=1)

    def test_a_series_too_short_for_the_embedding_is_rejected(self):
        with pytest.raises(ValueError, match="too short"):
            cross_map_curve(np.zeros(12), np.zeros(12), dimension=6, delay=3)

    def test_skill_spread_is_reported_so_differences_can_be_judged(self):
        x, y = coupled_logistic()
        result = cross_map_curve(x, y, dimension=3, delay=1, n_draws=6)
        assert result.skill_spread.shape == result.skill.shape
        assert np.all(result.skill_spread >= 0.0)

    def test_it_is_exported(self):
        assert hasattr(ts2net, "ccm")
        assert hasattr(ts2net, "ccm_test")
