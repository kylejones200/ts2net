#!/usr/bin/env python3
"""
Healthcare recipe — patient trajectory states via dynamic visibility graphs.

Synthetic vitals with a late risk shift.
Run: python examples/recipes/healthcare_trajectory.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np

from ts2net.reports import build_decision_package


def main() -> None:
    rng = np.random.default_rng(21)
    n = 240
    hr = 72 + rng.normal(0, 3, n)
    hr[160:] += np.linspace(0, 25, n - 160)
    spo2 = 98 - 0.05 * (hr - 72) + rng.normal(0, 0.3, n)

    pkg = build_decision_package(
        hr,
        method="hvg",
        window=35,
        step=3,
        title="Patient trajectory — heart rate graph shifts",
    )
    print(pkg.to_markdown())
    print()
    if pkg.dynamic_report:
        anom = pkg.dynamic_report.result.anomalous_windows()
        if len(anom):
            print(f"Review vitals at window indices: {anom.tolist()}")


if __name__ == "__main__":
    main()
