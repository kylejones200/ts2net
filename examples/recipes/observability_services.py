#!/usr/bin/env python3
"""
Observability recipe — service metrics dependency and incident precursors.

Synthetic latency panel; service B degrades and perturbs C.
Run: python examples/recipes/observability_services.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from ts2net.causal import CausalWorkflowSpec, run_causal_analysis
from ts2net.reports import build_decision_package


def main() -> None:
    rng = np.random.default_rng(12)
    n = 300
    a = np.abs(rng.normal(50, 5, n))
    b = 0.6 * a + rng.normal(0, 3, n)
    c = 0.4 * b + rng.normal(0, 2, n)
    b[200:] += np.linspace(0, 40, n - 200)
    c[220:] += np.linspace(0, 30, n - 220)

    X = np.vstack([a, b, c])
    names = ["api_gateway", "checkout_svc", "payment_db"]

    causal = run_causal_analysis(
        X,
        spec=CausalWorkflowSpec(method="granger", max_lag=3, series_names=names),
    )
    pkg = build_decision_package(
        b,
        method="hvg",
        window=30,
        causal=causal,
        title="Observability incident precursor",
    )
    print(pkg.to_markdown())


if __name__ == "__main__":
    main()
