#!/usr/bin/env python3
"""
Finance recipe — correlation network, regime change via dynamic graphs.

Synthetic returns with correlation breakdown mid-sample.
Run: python examples/recipes/finance_regime.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from ts2net.graphs import rolling_correlation_network
from ts2net.reports import build_decision_package


def main() -> None:
    rng = np.random.default_rng(99)
    n = 500
    # Two correlated assets; correlation drops after regime break
    r1 = rng.normal(0, 0.01, n)
    r2 = 0.8 * r1 + rng.normal(0, 0.006, n)
    r2[300:] = -0.5 * r1[300:] + rng.normal(0, 0.012, n - 300)
    X = np.column_stack([r1, r2])

    # Rolling correlation on asset 1 as instability proxy
    pkg = build_decision_package(
        r1,
        method="hvg",
        window=40,
        step=2,
        title="Finance regime — asset 1 rolling graph",
    )
    print(pkg.to_markdown())
    print()
    print("--- Rolling correlation network (last window) ---")
    G = rolling_correlation_network(X[-80:], threshold=0.3)
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
