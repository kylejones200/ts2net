"""
PySINDy dynamics discovery example.

Reproduces the linear 2D system from PySINDy tutorial 1 and builds a
coupling network from the discovered equations.

Run:
    pip install 'ts2net[sindy]'
    python examples/sindy_example.py

Reference:
    https://pysindy.readthedocs.io/en/latest/examples/tutorial_1/example.html
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts2net.sindy import SINDySpec, fit_sindy, sindy_coupling_network


def main() -> None:
    t = np.linspace(0, 1, 100)
    x0, y0 = 3.0, 0.5
    X = np.column_stack([x0 * np.exp(-2 * t), y0 * np.exp(t)])

    print("=" * 60)
    print("SINDy fit (tutorial 1 linear system)")
    print("=" * 60)
    result = fit_sindy(
        X,
        t,
        feature_names=["x", "y"],
        spec=SINDySpec(polynomial_degree=1, threshold=0.1),
    )
    for line in result.equations():
        print(f"  {line}")

    print()
    print("=" * 60)
    print("Coupling network (linear terms)")
    print("=" * 60)
    G = sindy_coupling_network(result, threshold=0.5, linear_only=True)
    for u, v, data in G.edges(data=True):
        print(f"  {u} -> {v}: weight={data['weight']:.3f}")

    print()
    print("=" * 60)
    print("Simulation check")
    print("=" * 60)
    t_sim = np.linspace(0, 1, 50)
    X_sim = result.simulate(X[0], t_sim)
    print(f"  Simulated shape: {X_sim.shape}")
    print(f"  x(0)={X_sim[0,0]:.3f}, y(0)={X_sim[0,1]:.3f}")


if __name__ == "__main__":
    main()
