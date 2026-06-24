#!/usr/bin/env python3
"""
Literature validation runner (Horizon 9 / v0.9).

Run:
    python benchmarks/run_validation.py
    TS2NET_CI_SMOKE=1 python benchmarks/run_validation.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.literature_checks import (  # noqa: E402
    run_all_fixtures,
    write_validation_manifest,
)


def main() -> int:
    smoke = os.environ.get("TS2NET_CI_SMOKE", "") == "1"
    results = run_all_fixtures(smoke=smoke)

    print("=" * 60)
    print("Literature validation" + (" (CI smoke)" if smoke else ""))
    print("=" * 60)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.id}: {r.message}")
        if r.citation:
            print(f"         {r.citation}")

    manifest_path = _REPO / "benchmarks" / "results" / "validation_manifest.json"
    payload = write_validation_manifest(results, manifest_path)
    print()
    print(f"Manifest written to {manifest_path}")
    print(f"Overall: {'PASS' if payload['passed'] else 'FAIL'}")

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
