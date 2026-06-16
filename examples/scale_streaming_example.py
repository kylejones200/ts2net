"""
Scale and streaming example (horizon 0.6).

Demonstrates window iterators, streaming stats, parallel build_windows,
sparse export, and performance contracts.

Run:
    python examples/scale_streaming_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ts2net import build_windows
from ts2net.scale import (
    build_windows_streaming,
    get_performance_contract,
    iter_windows,
    list_performance_contracts,
    stream_chunk_stats,
)


def main():
    rng = np.random.default_rng(42)
    n = 50_000
    x = np.cumsum(rng.standard_normal(n)) + 0.01 * rng.standard_normal(n)

    print("=" * 60)
    print("Performance contracts")
    print("=" * 60)
    for key in ("hvg", "nvg", "build_windows"):
        print(f"  {get_performance_contract(key).summary()}")

    print()
    print("=" * 60)
    print(f"Streaming windows (n={n:,}, window=48, step=24)")
    print("=" * 60)

    n_windows = 0
    for i, start, _ in iter_windows(x, width=48, step=24):
        n_windows += 1
        if i == 0:
            print(f"  First window starts at index {start}")
    print(f"  Total windows (iterator only): {n_windows}")

    print()
    print("=" * 60)
    print("Streaming graph stats (first 3 windows)")
    print("=" * 60)
    for i, start, stats in build_windows_streaming(x, window=48, step=24, method="hvg"):
        if i >= 3:
            break
        print(
            f"  window {i} @ {start}: "
            f"edges={stats['n_edges']}, avg_degree={stats['avg_degree']:.2f}"
        )

    print()
    print("=" * 60)
    print("Chunk-level HVG stats (stream_chunk_stats)")
    print("=" * 60)
    for i, stats in stream_chunk_stats(x, chunk_size=10_000, method="hvg"):
        if i >= 2:
            break
        print(
            f"  chunk {i}: edges={stats['n_edges']}, "
            f"avg_degree={stats['avg_degree']:.2f}"
        )

    print()
    print("=" * 60)
    print("Incremental HVG (streaming append)")
    print("=" * 60)
    from ts2net.scale import IncrementalHVG

    inc = IncrementalHVG()
    for v in x[:5]:
        r = inc.append(float(v))
        print(f"  append @{r.index}: +{len(r.new_edges)} edges (total {r.n_edges})")

    print()
    print("=" * 60)
    print("Parallel build_windows (n_jobs=2)")
    print("=" * 60)
    stats = build_windows(x, window=48, step=24, method="hvg", n_jobs=2)
    print(f"  Computed {len(stats['n_edges'])} window stat vectors")
    print(f"  Mean avg_degree: {np.nanmean(stats['avg_degree']):.2f}")

    print()
    print(f"Registered contracts: {', '.join(sorted(list_performance_contracts()))}")


if __name__ == "__main__":
    main()
