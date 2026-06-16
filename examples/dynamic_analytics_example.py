"""
Dynamic network analytics example (horizon 0.8).

Demonstrates regime detection, anomaly scoring, edge persistence,
community tracking, and the full run_dynamic_analysis workflow.

Run:
    python examples/dynamic_analytics_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts2net.dynamic import DynamicWorkflowSpec, run_dynamic_analysis


def main():
    rng = np.random.default_rng(7)
    n = 600
    t = np.arange(n)

    # Three regimes: low variance, high variance spike, return to baseline
    x = 0.2 * np.sin(2 * np.pi * t / 48) + 0.05 * rng.standard_normal(n)
    x[200:350] = x[200:350] + 1.5 * rng.standard_normal(150)
    x[450:520] += 4.0 * np.sin(2 * np.pi * t[450:520] / 6)

    print("=" * 60)
    print("Dynamic network analysis")
    print("=" * 60)

    spec = DynamicWorkflowSpec(
        method="hvg",
        window=48,
        step=12,
        regime_metric="avg_degree",
        regime_threshold=2.0,
    )
    result = run_dynamic_analysis(x, spec)

    print(result.summary())

    print()
    anomalous = result.anomalous_windows(threshold=1.5).tolist()[:10]
    print("Top anomalous windows:", anomalous)

    pers = sorted(result.persistence.items(), key=lambda kv: -kv[1])[:5]
    if pers:
        print()
        print("Most persistent edges:")
        for (u, v), score in pers:
            print(f"  ({u}, {v}): {score:.2f}")


if __name__ == "__main__":
    main()
