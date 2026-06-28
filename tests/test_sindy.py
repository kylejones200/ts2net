"""Tests for PySINDy integration."""

from __future__ import annotations

import numpy as np
import pytest

pysindy = pytest.importorskip("pysindy")

from ts2net.sindy import SINDySpec, fit_sindy, sindy_coupling_network


class TestSINDy:
    def test_linear_decoupled_system(self):
        """Recover dx/dt = -2x, dy/dt = y (PySINDy tutorial 1)."""
        t = np.linspace(0, 1, 80)
        X = np.column_stack([3.0 * np.exp(-2 * t), 0.5 * np.exp(t)])
        result = fit_sindy(
            X,
            t,
            feature_names=["x", "y"],
            spec=SINDySpec(polynomial_degree=1, threshold=0.1),
        )
        names = result.feature_names
        xi = names.index("x")
        yi = names.index("y")
        assert result.coefficients[0, xi] == pytest.approx(-2.0, abs=0.05)
        assert result.coefficients[1, yi] == pytest.approx(1.0, abs=0.05)

    def test_multiple_trajectories(self):
        t1 = np.linspace(0, 1, 50)
        t2 = np.linspace(0, 2, 80)
        X1 = np.column_stack([np.exp(-2 * t1), np.exp(t1)])
        X2 = np.column_stack([2 * np.exp(-2 * t2), 3 * np.exp(t2)])
        result = fit_sindy(
            [X1, X2],
            [t1, t2],
            feature_names=["x", "y"],
            spec=SINDySpec(polynomial_degree=1, threshold=0.1),
        )
        names = result.feature_names
        assert result.coefficients[0, names.index("x")] == pytest.approx(-2.0, abs=0.05)
        assert result.coefficients[1, names.index("y")] == pytest.approx(1.0, abs=0.05)

    def test_coupling_network_linear(self):
        t = np.linspace(0, 1, 60)
        X = np.column_stack([np.exp(-2 * t), np.exp(t)])
        result = fit_sindy(
            X,
            t,
            feature_names=["x", "y"],
            spec=SINDySpec(polynomial_degree=1, threshold=0.1),
        )
        G = sindy_coupling_network(result, threshold=0.5, linear_only=True)
        assert list(G.nodes()) == ["x", "y"]
        assert G.number_of_edges() == 0

    def test_simulate(self):
        t = np.linspace(0, 0.5, 40)
        X = np.column_stack([np.exp(-2 * t), np.exp(t)])
        result = fit_sindy(
            X,
            t,
            feature_names=["x", "y"],
            spec=SINDySpec(polynomial_degree=1, threshold=0.1),
        )
        sim = result.simulate(np.array([1.0, 1.0]), np.linspace(0, 0.5, 20))
        assert sim.shape == (20, 2)

    def test_equations_string(self):
        t = np.linspace(0, 1, 40)
        X = np.column_stack([np.exp(-2 * t), np.exp(t)])
        result = fit_sindy(X, t, feature_names=["x", "y"], spec=SINDySpec(polynomial_degree=1))
        eqs = result.equations()
        assert len(eqs) == 2
        assert "(x)'" in eqs[0]
