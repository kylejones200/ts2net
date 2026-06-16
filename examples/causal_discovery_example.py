"""
Constraint-based causal discovery example (milestone 0.9).

Demonstrates PC and FCI on synthetic data and lag-expanded time series.

Run:
    python examples/causal_discovery_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts2net.causal import (
    pc_algorithm,
    fci_algorithm,
    pc_timeseries_network,
    fci_timeseries_network,
)


def main() -> None:
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PC: collider X -> Z <- Y")
    print("=" * 60)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)
    z = 0.85 * x + 0.85 * y + 0.05 * rng.standard_normal(2000)
    pc_result = pc_algorithm(
        np.column_stack([x, y, z]),
        alpha=0.01,
        variable_names=["X", "Y", "Z"],
    )
    print(f"Skeleton edges: {list(pc_result.skeleton.edges())}")
    print(f"Oriented edges: {list(pc_result.cpdag.edges(data=True))}")

    print()
    print("=" * 60)
    print("FCI: latent confounder between X and Y")
    print("=" * 60)
    lat = rng.standard_normal(2000)
    x2 = lat + 0.1 * rng.standard_normal(2000)
    y2 = lat + 0.1 * rng.standard_normal(2000)
    fci_result = fci_algorithm(np.column_stack([x2, y2]), alpha=0.05)
    for u, v, data in fci_result.pag.edges(data=True):
        print(f"  {fci_result.variable_names[u]} — {fci_result.variable_names[v]}: {data.get('edge_label')}")

    print()
    print("=" * 60)
    print("PC on lag-expanded time series")
    print("=" * 60)
    n = 500
    driver = rng.standard_normal(n)
    response = np.concatenate([[0], 0.7 * driver[:-1] + 0.05 * rng.standard_normal(n - 1)])
    ts_result = pc_timeseries_network(
        [driver, response],
        max_lag=2,
        series_names=["driver", "response"],
        alpha=0.05,
    )
    print(f"Variables: {ts_result.variable_names}")
    print(f"Skeleton edges: {ts_result.skeleton.number_of_edges()}")


if __name__ == "__main__":
    main()
