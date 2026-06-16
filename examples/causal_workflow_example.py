"""
Causal network workflow example (horizon 0.5).

Demonstrates lag search, edge confidence, confounder adjustment,
and plain-language causal summaries.

Run:
    python examples/causal_workflow_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts2net.causal import run_causal_analysis, CausalWorkflowSpec


def main():
    rng = np.random.default_rng(42)

    # Three-series panel: x1 drives x2; x3 is independent noise
    n = 500
    x1 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    x2 = np.concatenate([[0], 0.6 * x1[:-1] + 0.05 * rng.standard_normal(n - 1)])

    names = ["sensor_A", "sensor_B", "sensor_C"]

    print("=" * 60)
    print("Granger workflow (lag search + summary)")
    print("=" * 60)
    granger_result = run_causal_analysis(
        [x1, x2, x3],
        spec=CausalWorkflowSpec(
            method="granger",
            lag_search=True,
            max_lag=5,
            alpha=0.05,
            series_names=names,
        ),
    )
    print(granger_result.summary())
    print(f"Significant edges: {len(granger_result.significant_edges())}")

    print()
    print("=" * 60)
    print("Transfer entropy workflow (permutation confidence)")
    print("=" * 60)
    te_result = run_causal_analysis(
        [x1, x2],
        method="transfer_entropy",
        lag_search=True,
        te_lags=[1, 2, 3],
        n_permutations=49,
        bins=8,
        series_names=["sensor_A", "sensor_B"],
    )
    print(te_result.summary())

    print()
    print("=" * 60)
    print("Confounder-adjusted Granger")
    print("=" * 60)
    # x2 depends on x1 and x3; test x1 -> x2 controlling for x3
    x2_conf = np.concatenate(
        [[0], 0.5 * x1[:-1] + 0.4 * x3[:-1] + 0.05 * rng.standard_normal(n - 1)]
    )
    adj_result = run_causal_analysis(
        [x1, x2_conf, x3],
        method="granger",
        lag_search=True,
        adjust_confounders=True,
        max_lag=4,
        series_names=names,
    )
    print(adj_result.summary())


if __name__ == "__main__":
    main()
