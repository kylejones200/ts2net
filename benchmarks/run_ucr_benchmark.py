#!/usr/bin/env python3
"""
UCR classification benchmark harness (Horizon 9 / v0.9).

Run:
    python benchmarks/run_ucr_benchmark.py
    TS2NET_CI_SMOKE=1 python benchmarks/run_ucr_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ts2net.datasets.ucr import run_ucr_benchmark  # noqa: E402


def main() -> int:
    smoke = os.environ.get("TS2NET_CI_SMOKE", "") == "1"
    dataset = "GunPoint" if not smoke else "GunPoint"
    output = _REPO / "benchmarks" / "results" / "ucr_benchmark.json"

    payload = run_ucr_benchmark(
        dataset,
        cv=3 if smoke else 5,
        include_optional_baselines=not smoke,
        output_path=output,
    )

    print("=" * 60)
    print("UCR benchmark" + (" (CI smoke)" if smoke else ""))
    print("=" * 60)
    print(f"Dataset: {payload['dataset']} (source: {payload['metadata']['source']})")
    for name, scores in payload["scores"].items():
        print(
            f"  {name}: {scores['mean_score']:.3f} ± {scores['std_score']:.3f} "
            f"({scores['n_features']} features)"
        )
    print(f"\nResults written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
