#!/usr/bin/env python3
"""
Energy production recipe — similarity graph, abnormal decline flag.

Uses synthetic well decline curves; flags a well with accelerated decline.
Run: python examples/recipes/energy_production.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from ts2net.graphs import similarity_network
from ts2net.reports import build_graph_report, build_decision_package


def main() -> None:
    rng = np.random.default_rng(7)
    n_wells, n_months = 8, 60
    t = np.arange(n_months)
    curves = []
    names = []
    for i in range(n_wells):
        q0 = 100 + 10 * i
        decline = 0.02 + 0.002 * i
        if i == 3:
            decline = 0.08  # abnormal well
        q = q0 * np.exp(-decline * t) + rng.normal(0, 2, n_months)
        curves.append(q)
        names.append(f"well_{i}")

    X = np.vstack(curves)
    G, _ = similarity_network(X, method="dtw", threshold=0.35)
    for i, name in enumerate(names):
        if G.has_node(i):
            G.nodes[i]["name"] = name

    report = build_graph_report(
        G,
        method="similarity_dtw",
        parameters={"threshold": 0.35},
        title="Well analog network",
    )
    print(report.summary())
    print()

    # Flag abnormal well via degree isolation
    deg = dict(G.degree())
    isolated = [names[n] for n, d in deg.items() if d == 0]
    if isolated:
        print(f"Wells with no analogs (review decline): {isolated}")

    # Dynamic check on abnormal well series
    pkg = build_decision_package(
        curves[3],
        G=G,
        method="hvg",
        window=20,
        title="Abnormal decline well — decision package",
    )
    print()
    print(pkg.summary())


if __name__ == "__main__":
    main()
