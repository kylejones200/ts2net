#!/usr/bin/env python3
"""
Industrial sensor recipe — drift, causal drivers, failure precursors.

Synthetic 3-sensor panel with injected drift on sensor 2.
Run: python examples/recipes/industrial_sensors.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.reports import build_decision_package


def main() -> None:
    rng = np.random.default_rng(42)
    n = 400
    t = np.arange(n)
    # Sensor 0 drives 1; sensor 2 drifts after t=250
    s0 = np.cumsum(rng.normal(0, 0.5, n))
    s1 = 0.7 * np.roll(s0, 2) + rng.normal(0, 0.2, n)
    s2 = rng.normal(0, 0.3, n)
    s2[250:] += np.linspace(0, 3, n - 250)

    X = np.vstack([s0, s1, s2])
    names = ["motor_vibration", "bearing_temp", "line_pressure"]

    causal = run_causal_analysis(
        X,
        spec=CausalWorkflowSpec(method="granger", max_lag=5, alpha=0.05, series_names=names),
    )

    pkg = build_decision_package(
        x=s2,
        method="hvg",
        window=50,
        causal=causal,
        title="Industrial sensor decision package",
    )
    print(pkg.to_markdown())


if __name__ == "__main__":
    main()
