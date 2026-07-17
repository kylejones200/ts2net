#!/usr/bin/env python3
"""
Finance recipe — bundled FRED-style macro panel.

Uses offline monthly indicators (SP500 returns proxy, VIX, unemployment).
Run: python examples/recipes/finance_fred_real.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.datasets import load_dataset
from ts2net.graphs import correlation_network
from ts2net.reports import build_decision_package, build_graph_report


def main() -> None:
    panel = load_dataset("fred_macro_panel")
    X = panel["X"]
    names = panel["metadata"]["series_names"]
    print(f"Loaded {len(names)} macro series, {panel['metadata']['n_points']} months")
    print(f"Series: {', '.join(names)}")
    print()

    G, _, _ = correlation_network(X, threshold=0.35, rule="threshold")
    for i, name in enumerate(names):
        if G.has_node(i):
            G.nodes[i]["name"] = name

    report = build_graph_report(
        G,
        method="correlation",
        parameters={"threshold": 0.35},
        title="Macro correlation network",
    )
    print(report.summary())
    print()

    causal = run_causal_analysis(
        X,
        spec=CausalWorkflowSpec(
            method="granger",
            max_lag=3,
            alpha=0.05,
            series_names=names,
        ),
    )

    pkg = build_decision_package(
        X[0],
        G=G,
        method="hvg",
        window=12,
        step=2,
        causal=causal,
        title="Macro regime — SP500 returns graph shifts",
    )
    print(pkg.to_markdown())


if __name__ == "__main__":
    main()
