"""Contracts for the reconstructed state space.

`reconstruct` owns the delay-embedding convention, chooses a delay and a
dimension by named rules, and reports when it could not. These assert against
systems whose embedding dimension is known analytically rather than against
recorded output, and check that the failure to find a reconstruction is
reported rather than papered over.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import ts2net
from ts2net.state import FNN_TOLERANCE, embed, reconstruct


def lorenz_x(n: int = 6000, dt: float = 0.01, transient: int = 1000) -> np.ndarray:
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    state = np.array([1.0, 1.0, 1.0])
    out = []
    for i in range(n + transient):
        x, y, z = state
        state = state + dt * np.array(
            [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
        )
        if i >= transient:
            out.append(state[0])
    return np.asarray(out)


def sine(n: int = 4000, step: float = 0.05) -> np.ndarray:
    return np.sin(np.arange(n) * step)


def white_noise(n: int = 4000) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n)


class TestEmbeddingConvention:
    """One convention, owned here, matching what the package already used."""

    @pytest.mark.parametrize("dimension, delay", [(2, 1), (3, 4), (5, 2), (1, 7)])
    def test_shape_and_column_meaning(self, dimension, delay):
        x = np.arange(200.0)
        points = embed(x, dimension, delay)
        assert points.shape == (len(x) - (dimension - 1) * delay, dimension)
        for column in range(dimension):
            offset = column * delay
            np.testing.assert_array_equal(
                points[:, column], x[offset : offset + points.shape[0]]
            )

    def test_a_row_is_one_state_vector(self):
        points = embed(np.arange(20.0), 3, 2)
        np.testing.assert_array_equal(points[0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(points[5], [5.0, 7.0, 9.0])

    def test_it_agrees_with_the_existing_implementation(self):
        from ts2net.core import embed as core_embed

        x = sine(500)
        for dimension, delay in [(2, 1), (4, 5), (3, 9)]:
            np.testing.assert_array_equal(
                embed(x, dimension, delay), core_embed(x, dimension, delay)
            )

    @pytest.mark.parametrize("dimension, delay", [(0, 1), (2, 0), (-1, 1)])
    def test_degenerate_parameters_are_rejected(self, dimension, delay):
        with pytest.raises(ValueError, match="at least 1"):
            embed(sine(100), dimension, delay)

    def test_too_short_a_series_is_rejected_with_the_numbers(self):
        with pytest.raises(ValueError, match="too short"):
            embed(np.arange(10.0), dimension=5, delay=4)


class TestParameterSelection:
    def test_lorenz_reconstructs_in_three_dimensions(self):
        state = reconstruct(lorenz_x())
        assert state.embedding_dimension == 3
        assert state.diagnostics.dimension_converged
        assert 8 <= state.delay <= 25

    def test_a_sine_reconstructs_in_two(self):
        state = reconstruct(sine())
        assert state.embedding_dimension == 2
        assert state.diagnostics.dimension_converged

    def test_the_chosen_dimension_is_the_first_below_tolerance(self):
        state = reconstruct(lorenz_x())
        curve = state.diagnostics.fnn_curve
        chosen = state.embedding_dimension
        assert curve[chosen - 1] <= FNN_TOLERANCE
        assert np.all(curve[: chosen - 1] > FNN_TOLERANCE)

    def test_explicit_parameters_are_honoured(self):
        state = reconstruct(lorenz_x(), dimension=5, delay=7)
        assert state.embedding_dimension == 5
        assert state.delay == 7
        assert state.points.shape[1] == 5

    def test_the_delay_rule_is_recorded_and_changes_the_answer(self):
        series = lorenz_x()
        mutual = reconstruct(series, delay_rule="mutual_information")
        linear = reconstruct(series, delay_rule="autocorrelation", max_lag=300)
        assert mutual.diagnostics.delay_rule == "mutual_information"
        assert linear.diagnostics.delay_rule == "autocorrelation"
        assert linear.delay > mutual.delay, (
            "the two rules disagree on a chaotic attractor; if they stop "
            "disagreeing this test should be re-derived, not deleted"
        )


class TestItSaysWhenThereIsNoReconstruction:
    """The failure mode this project has now met four times."""

    def test_white_noise_reports_no_finite_embedding_dimension(self):
        state = reconstruct(white_noise())
        assert not state.diagnostics.dimension_converged
        assert state.diagnostics.looks_stochastic
        assert "no finite embedding dimension" in state.diagnostics.summary()

    def test_a_deterministic_signal_is_not_called_stochastic(self):
        for series in (lorenz_x(), sine()):
            assert not reconstruct(series).diagnostics.looks_stochastic

    def test_the_fallback_dimension_is_flagged_not_hidden(self):
        state = reconstruct(white_noise(), max_dimension=6)
        assert state.embedding_dimension == 6
        assert not state.diagnostics.dimension_converged

    def test_an_unmeasured_delay_is_flagged(self):
        state = reconstruct(white_noise())
        assert not state.diagnostics.delay_converged
        assert "delay NOT measured" in state.diagnostics.summary()


class TestRecurrenceView:
    def test_adjacency_is_square_symmetric_and_hollow(self):
        state = reconstruct(lorenz_x(2000))
        epsilon = state.epsilon_for_density(0.05)
        adjacency = np.asarray(state.recurrence(epsilon))
        n = len(state)
        assert adjacency.shape == (n, n)
        np.testing.assert_array_equal(adjacency, adjacency.T)
        assert np.all(np.diag(adjacency) == 0)

    def test_epsilon_for_density_hits_the_requested_density(self):
        state = reconstruct(lorenz_x(1500))
        for target in (0.02, 0.05, 0.15):
            adjacency = np.asarray(state.recurrence(state.epsilon_for_density(target)))
            n = adjacency.shape[0]
            density = adjacency.sum() / (n * (n - 1))
            assert density == pytest.approx(target, abs=0.02), target

    def test_a_larger_epsilon_never_removes_an_edge(self):
        state = reconstruct(lorenz_x(1200))
        small = np.asarray(state.recurrence(state.epsilon_for_density(0.05)))
        large = np.asarray(state.recurrence(state.epsilon_for_density(0.15)))
        assert np.all(large >= small)

    def test_the_theiler_window_removes_near_diagonal_recurrence(self):
        state = reconstruct(lorenz_x(1200))
        epsilon = state.epsilon_for_density(0.05)
        without = np.asarray(state.recurrence(epsilon)).sum()
        with_window = np.asarray(state.recurrence(epsilon, theiler=10)).sum()
        assert with_window < without

    def test_the_network_view_indexes_nodes_by_time(self):
        state = reconstruct(lorenz_x(800))
        graph = state.recurrence_network(state.epsilon_for_density(0.05))
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == len(state)
        assert set(graph.nodes()) == set(range(len(state)))
        assert nx.number_of_selfloops(graph) == 0

    def test_epsilon_must_be_positive_and_density_a_fraction(self):
        state = reconstruct(sine(600))
        with pytest.raises(ValueError, match="epsilon must be positive"):
            state.recurrence(0.0)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            state.epsilon_for_density(1.5)


class TestPublicSurface:
    def test_reconstruct_is_exported(self):
        assert hasattr(ts2net, "reconstruct")
        assert hasattr(ts2net, "StateSpace")

    def test_a_short_series_is_rejected(self):
        with pytest.raises(ValueError, match="at least 10 points"):
            reconstruct(np.arange(5.0))
