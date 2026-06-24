"""
Research validation example (Horizon 9).

Demonstrates the reference dataset registry and literature fixtures.

Run:
    python examples/validation_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.literature_checks import run_all_fixtures
from ts2net.datasets import list_datasets, load_dataset


def main() -> None:
    print("=" * 60)
    print("Reference datasets")
    print("=" * 60)
    for name in list_datasets():
        try:
            data = load_dataset(name, n=500, seed=0)
        except FileNotFoundError as exc:
            print(f"  {name}: skipped ({exc})")
            continue
        meta = data["metadata"]
        print(f"  {name}: task={meta['task']}, shape={data['X'].shape}")

    print()
    print("=" * 60)
    print("Literature fixtures (smoke subset)")
    print("=" * 60)
    for r in run_all_fixtures(smoke=True):
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.id}: {r.message}")


if __name__ == "__main__":
    main()
