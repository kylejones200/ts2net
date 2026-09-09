#!/usr/bin/env python3
"""Compare RBF bandwidth policies for `node_roles_spectral`.

Analysis only; changes no default.

`DEFAULT_SPECTRAL_GAMMA` is a fixed constant, the historical 1/12. The median
heuristic instead reads the spread the data actually has. This measures both
without a ground-truth clustering, using diagnostics that say whether a kernel
is informative at all:

* **kernel saturation.** As gamma -> 0 the affinity matrix tends to all ones and
  every node looks identical; as gamma -> infinity it tends to the identity and
  every node is isolated. Either way the spectral problem carries no
  information. Mean off-diagonal affinity near 0 or 1, or a near-zero spread of
  affinities, marks a badly scaled kernel.
* **seed stability.** A well-conditioned spectral problem returns the same
  partition from different random seeds. Instability indicates the embedding is
  not cleanly separable.
* **effective dimension.** For standardized features the expected squared
  distance is 2p over p *varying* columns, so a bandwidth of order 1/p is
  reasonable -- but p should be the number of columns that actually vary, not
  the matrix width. This reports both.

Run:  uv run python scripts/experiment_spectral_bandwidth.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel

import ts2net  # noqa: F401
from ts2net.networks import roles

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_role_features import corpus  # noqa: E402

warnings.filterwarnings("ignore")


def features(graph):
    return roles.role_features_extended(graph)[1]


def varying_columns(mat: np.ndarray) -> int:
    return int(np.sum(mat.std(axis=0, ddof=1) > 1e-9)) if mat.shape[0] > 1 else 0


def kernel_stats(mat: np.ndarray, gamma: float) -> dict:
    affinity = rbf_kernel(mat, gamma=gamma)
    off = affinity[np.triu_indices_from(affinity, k=1)]
    return {
        "mean": float(np.mean(off)),
        "std": float(np.std(off)),
        "frac_above_0.99": float(np.mean(off > 0.99)),
        "frac_below_0.01": float(np.mean(off < 0.01)),
    }


def seed_stability(mat: np.ndarray, gamma: float, n_roles: int, seeds) -> float:
    labels = []
    for seed in seeds:
        affinity = rbf_kernel(mat, gamma=gamma)
        labels.append(
            SpectralClustering(
                n_clusters=n_roles,
                affinity="precomputed",
                random_state=seed,
                assign_labels="kmeans",
                n_init=20,
            ).fit_predict(affinity)
        )
    pairs = [
        adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)
    ]
    return float(np.median(pairs)) if pairs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-roles", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[3363, 7, 42, 101])
    args = ap.parse_args()

    graphs = corpus()

    print("=" * 96)
    print("1. BANDWIDTH VALUES  (fixed 1/12 vs median heuristic)")
    print("=" * 96)
    print(
        f"{'graph':<18} {'n':>4} {'cols':>5} {'varying':>8} "
        f"{'gamma_fixed':>12} {'gamma_median':>13} {'ratio':>7}"
    )
    rows = {}
    for name, graph in graphs.items():
        mat = features(graph)
        gm = roles.median_heuristic_gamma(mat)
        gf = roles.DEFAULT_SPECTRAL_GAMMA
        rows[name] = (mat, gf, gm)
        print(
            f"{name:<18} {mat.shape[0]:>4} {mat.shape[1]:>5} {varying_columns(mat):>8} "
            f"{gf:>12.5f} {gm:>13.5f} {gm / gf:>7.2f}"
        )

    print()
    print("=" * 96)
    print("2. KERNEL SATURATION  (mean off-diagonal affinity; 0 or 1 is uninformative)")
    print("=" * 96)
    print(
        f"{'graph':<18} {'fixed mean':>11} {'fixed std':>10} {'>0.99':>7} "
        f"| {'median mean':>12} {'median std':>11} {'>0.99':>7}"
    )
    for name, (mat, gf, gm) in rows.items():
        if mat.shape[0] < 2:
            continue
        a, b = kernel_stats(mat, gf), kernel_stats(mat, gm)
        print(
            f"{name:<18} {a['mean']:>11.4f} {a['std']:>10.4f} "
            f"{a['frac_above_0.99']:>7.2f} | {b['mean']:>12.4f} "
            f"{b['std']:>11.4f} {b['frac_above_0.99']:>7.2f}"
        )

    print()
    print("=" * 96)
    print("3. SEED STABILITY  (median pairwise ARI over seeds; 1.0 = fully stable)")
    print("=" * 96)
    print(f"{'graph':<18} {'fixed':>8} {'median':>8}  verdict")
    fixed_all, median_all = [], []
    for name, (mat, gf, gm) in rows.items():
        if mat.shape[0] <= args.n_roles or np.allclose(mat, 0):
            print(f"{name:<18} {'-':>8} {'-':>8}  skipped (degenerate)")
            continue
        sf = seed_stability(mat, gf, args.n_roles, args.seeds)
        sm = seed_stability(mat, gm, args.n_roles, args.seeds)
        fixed_all.append(sf)
        median_all.append(sm)
        verdict = "median better" if sm > sf + 1e-9 else (
            "fixed better" if sf > sm + 1e-9 else "tie"
        )
        print(f"{name:<18} {sf:>8.3f} {sm:>8.3f}  {verdict}")
    if fixed_all:
        print()
        print(
            f"  aggregate  fixed median {np.median(fixed_all):.3f} mean "
            f"{np.mean(fixed_all):.3f} | median-heuristic median "
            f"{np.median(median_all):.3f} mean {np.mean(median_all):.3f}"
        )
        wins = sum(1 for f, m in zip(fixed_all, median_all) if m > f + 1e-9)
        losses = sum(1 for f, m in zip(fixed_all, median_all) if f > m + 1e-9)
        print(
            f"  median heuristic more stable on {wins}, less on {losses}, "
            f"tied on {len(fixed_all) - wins - losses}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
