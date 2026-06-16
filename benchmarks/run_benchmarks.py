#!/usr/bin/env python3
"""
ts2net performance benchmarks.

Measures wall-clock time and (optionally) peak memory across all four
graph builders at a range of series lengths.  Outputs a Markdown table
to stdout and writes raw CSV data to benchmarks/results/.

Usage
-----
    python benchmarks/run_benchmarks.py [--repeats N] [--output-dir DIR]

    --repeats N     Number of timed repetitions per (method, n) pair (default 3).
    --output-dir    Directory for CSV output (default: benchmarks/results/).

Environment
-----------
    TS2NET_SKIP_LARGE=1   Skip n > 100 000 (useful for CI).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork
from ts2net.distances.dtw import cdist_dtw, _BACKEND as DTW_BACKEND


# ── configuration ──────────────────────────────────────────────────────────────

SERIES_LENGTHS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
DTW_CONFIGS = [
    (5,   100),
    (10,  100),
    (20,  100),
    (50,  100),
    (100, 100),
    (20,  500),
    (20,  2_000),
]

SKIP_LARGE = os.environ.get("TS2NET_SKIP_LARGE", "0") == "1"
CI_SMOKE = os.environ.get("TS2NET_CI_SMOKE", "0") == "1"
LARGE_THRESHOLD = 100_000

if CI_SMOKE:
    SERIES_LENGTHS = [100, 1_000, 5_000]
    DTW_CONFIGS = [(5, 50)]
    RECURRENCE_LENGTHS = [100, 500, 1_000]
else:
    RECURRENCE_LENGTHS = [100, 500, 1_000, 5_000, 10_000]


# ── timing helpers ─────────────────────────────────────────────────────────────

def time_ms(fn: Callable, repeats: int = 3) -> float:
    """Return median wall-clock time in milliseconds over `repeats` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ── benchmark suites ──────────────────────────────────────────────────────────

def bench_visibility(rng: np.random.Generator, repeats: int) -> list[dict]:
    """HVG and NVG at all series lengths."""
    rows = []
    for n in SERIES_LENGTHS:
        if SKIP_LARGE and n > LARGE_THRESHOLD:
            continue
        x = rng.standard_normal(n)
        # NVG needs a horizon limit for large series (O(n²) without it)
        nvg_limit = min(n, 2_000) if n > 10_000 else None
        nvg_kwargs = {"limit": nvg_limit} if nvg_limit else {}

        for method, fn, note in [
            ("HVG",          lambda x=x: HVG().build(x),                          ""),
            ("HVG[deg]",     lambda x=x: HVG(output="degrees").build(x),          "degree-only mode"),
            *(
                []
                if CI_SMOKE
                else [("HVG[directed]", lambda x=x: HVG(directed=True).build(x), "")]
            ),
            ("NVG",          lambda x=x, kw=nvg_kwargs: NVG(**kw).build(x),       f"limit={nvg_limit or 'none'}"),
        ]:
            ms = time_ms(fn, repeats)
            rows.append({"method": method, "n": n, "ms": ms, "note": note})
            print(f"  {method:<16} n={n:>8,}  {ms:8.1f} ms  {note}")

    return rows


def bench_recurrence(rng: np.random.Generator, repeats: int) -> list[dict]:
    """RecurrenceNetwork (knn and epsilon) at smaller series lengths."""
    rows = []
    for n in RECURRENCE_LENGTHS:
        if SKIP_LARGE and n > LARGE_THRESHOLD:
            continue
        x = rng.standard_normal(n)
        configs = [
            ("RN(knn,k=5)",   lambda x=x: RecurrenceNetwork(rule="knn", k=5).build(x), ""),
            ("RN(knn)[deg]",  lambda x=x: RecurrenceNetwork(rule="knn", k=5, output="degrees").build(x), "degree-only"),
        ]
        if not CI_SMOKE or n <= 500:
            configs.insert(
                1,
                ("RN(eps,ε=0.3)", lambda x=x: RecurrenceNetwork(rule="epsilon", epsilon=0.3).build(x), ""),
            )
        for method, fn, note in configs:
            ms = time_ms(fn, repeats)
            rows.append({"method": method, "n": n, "ms": ms, "note": note})
            print(f"  {method:<20} n={n:>7,}  {ms:8.1f} ms  {note}")

    return rows


def bench_transition(rng: np.random.Generator, repeats: int) -> list[dict]:
    """TransitionNetwork at all series lengths."""
    rows = []
    for n in SERIES_LENGTHS:
        if SKIP_LARGE and n > LARGE_THRESHOLD:
            continue
        x = rng.standard_normal(n)
        for order in [2, 3]:
            fn = lambda x=x, o=order: TransitionNetwork(symbolizer="ordinal", order=o).build(x)
            ms = time_ms(fn, repeats)
            method = f"TN(ord,order={order})"
            rows.append({"method": method, "n": n, "ms": ms, "note": ""})
            print(f"  {method:<22} n={n:>8,}  {ms:8.1f} ms")

    return rows


def bench_dtw(rng: np.random.Generator, repeats: int) -> list[dict]:
    """cdist_dtw (pairwise DTW) at various (k_series, series_length) configs."""
    rows = []
    print(f"  DTW backend: {DTW_BACKEND}")
    for k, L in DTW_CONFIGS:
        X = rng.standard_normal((k, L))
        ms = time_ms(lambda X=X: cdist_dtw(X), repeats)
        rows.append({"method": "cdist_dtw", "k": k, "L": L, "ms": ms})
        print(f"  cdist_dtw  {k:4d} series × {L:5d} pts  {ms:8.1f} ms  ({k*(k-1)//2} pairs)")

    return rows


# ── markdown table builder ─────────────────────────────────────────────────────

def _fmt(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f} µs"
    if ms < 1_000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.1f} s"


def build_visibility_table(rows: list[dict]) -> str:
    methods = ["HVG", "HVG[deg]", "NVG"]
    ns = sorted({r["n"] for r in rows if r["method"] in methods})
    by = {(r["method"], r["n"]): r["ms"] for r in rows}

    header = "| n |" + "".join(f" {m} |" for m in methods)
    sep    = "|---|" + "".join("---|" for _ in methods)
    lines  = [header, sep]
    for n in ns:
        vals = "".join(
            f" {_fmt(by[(m,n)]) if (m,n) in by else '—'} |"
            for m in methods
        )
        lines.append(f"| {n:,} |{vals}")
    return "\n".join(lines)


def build_dtw_table(rows: list[dict]) -> str:
    header = "| series (k) | length (L) | pairs | time |"
    sep    = "|---|---|---|---|"
    lines  = [header, sep]
    for r in rows:
        k, L, ms = r["k"], r["L"], r["ms"]
        lines.append(f"| {k} | {L} | {k*(k-1)//2} | {_fmt(ms)} |")
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats",    type=int,  default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print(f"ts2net benchmarks  ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"  Python {sys.version.split()[0]}  DTW backend: {DTW_BACKEND}")
    print(f"  repeats={args.repeats}  SKIP_LARGE={SKIP_LARGE}  CI_SMOKE={CI_SMOKE}")
    print("=" * 60)

    print("\n── Visibility graphs (HVG / NVG) ──")
    vis_rows = bench_visibility(rng, args.repeats)

    print("\n── Recurrence network ──")
    rn_rows = bench_recurrence(rng, args.repeats)

    print("\n── Transition network ──")
    tn_rows = bench_transition(rng, args.repeats)

    print("\n── DTW pairwise distance ──")
    dtw_rows = bench_dtw(rng, args.repeats) if not CI_SMOKE else []
    if CI_SMOKE:
        print("  (skipped in CI smoke mode)")

    # Write CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"benchmark_{ts}.csv"
    all_rows = vis_rows + rn_rows + tn_rows
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n", "ms", "note"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV → {csv_path}")

    # Print Markdown
    print("\n" + "=" * 60)
    print("MARKDOWN TABLES")
    print("=" * 60)
    print("\n### Visibility graphs\n")
    print(build_visibility_table(vis_rows))
    print(f"\n> NVG uses `limit=2000` for n > 10 000 (unconstrained NVG is O(n²)).")
    print("\n### DTW pairwise (cdist_dtw)\n")
    if dtw_rows:
        print(build_dtw_table(dtw_rows))
        print(f"\n> Backend: `{DTW_BACKEND}`.  Install with `pip install ts2net` for the Rust backend.")
    else:
        print("_Skipped in CI smoke mode._")


if __name__ == "__main__":
    main()
