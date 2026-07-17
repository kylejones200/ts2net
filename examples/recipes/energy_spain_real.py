#!/usr/bin/env python3
"""
Energy recipe — real Spain smart-meter summary panel.

Uses bundled per-meter network statistics from the Spain multi-meter experiment.
Run: python examples/recipes/energy_spain_real.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from ts2net.datasets import load_dataset
from ts2net.datasets.ucr import load_ucr
from ts2net.graphs import similarity_network
from ts2net.reports import build_decision_package, build_graph_report


def main() -> None:
    panel = load_dataset("spain_meters_summary")
    X = panel["X"]
    meta = panel["metadata"]
    print(f"Loaded {meta['n_meters']} Spain meters from bundled summary CSV")
    print(f"Features: {', '.join(meta['feature_cols'])}")
    print()

    G, _ = similarity_network(X, method="euclidean", threshold=0.25)
    report = build_graph_report(
        G,
        method="similarity_euclidean",
        parameters={"threshold": 0.25},
        title="Spain meter analog network (network metrics)",
    )
    print(report.summary())
    print()

    isolated = [n for n, d in G.degree() if d == 0]
    if isolated:
        print(f"Meters with no analogs (review consumption pattern): {len(isolated)}")
        print()

    # Real univariate series: Italy power demand (bundled UCR archive)
    X_ucr, _ = load_ucr("ItalyPowerDemand", split="train")
    series = X_ucr[0]
    pkg = build_decision_package(
        series,
        G=G,
        method="hvg",
        window=24,
        title="Italy power demand — rolling graph decision package",
    )
    print(pkg.to_markdown())


if __name__ == "__main__":
    main()
