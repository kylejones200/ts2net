#!/usr/bin/env python3
"""
Standalone DecisionPackage walkthrough.

Shows graph report, dynamic change detection, and causal evidence in one package.
Run: python examples/decision_package_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from ts2net import HVG
from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.reports import build_decision_package, build_graph_report


def main() -> None:
    rng = np.random.default_rng(0)
    n = 300
    driver = np.cumsum(rng.normal(0, 0.3, n))
    response = 0.6 * np.roll(driver, 1) + rng.normal(0, 0.15, n)
    signal = driver.copy()
    signal[180:] += np.linspace(0, 2.5, n - 180)

    print("=== Step 1: Snapshot graph report ===")
    b = HVG()
    b.build(signal)
    G = b.as_networkx()
    report = build_graph_report(G, method="hvg", n_points=n)
    print(report.summary())
    print()

    print("=== Step 2: Causal panel (driver → response) ===")
    X = np.vstack([driver, response, signal])
    causal = run_causal_analysis(
        X,
        spec=CausalWorkflowSpec(
            method="granger",
            max_lag=4,
            series_names=["driver", "response", "monitored"],
        ),
    )
    print(causal.summary())
    print()

    print("=== Step 3: Decision package (evidence + next actions) ===")
    pkg = build_decision_package(
        signal,
        G=G,
        method="hvg",
        window=40,
        step=2,
        causal=causal,
        title="Decision package walkthrough",
    )
    print(pkg.to_markdown())
    print()
    print("Structured fields:")
    print(f"  assumptions: {len(pkg.assumptions)}")
    print(f"  evidence: {len(pkg.evidence)}")
    print(f"  confidence: {len(pkg.confidence)}")
    print(f"  changes: {len(pkg.changes)}")
    print(f"  next_actions: {len(pkg.next_actions)}")


if __name__ == "__main__":
    main()
