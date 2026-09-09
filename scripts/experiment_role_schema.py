#!/usr/bin/env python3
"""Controlled comparison of role-feature schema v1 and v2, and of spectral gamma.

Analysis only. Nothing here changes production behaviour: `role_features_extended`
still returns the twelve-column v1 matrix, and `node_roles_spectral` still
derives gamma from the matrix width.

Switching from v1 (twelve columns) to v2 (nine) currently changes two things at
once, because `node_roles_spectral` sets ``gamma = 1 / X.shape[1]``. This runs
the full factorial so the schema effect, the gamma effect and their interaction
can be told apart:

    A = v1, gamma 1/12   production today
    B = v2, gamma 1/12   schema changed, bandwidth held
    C = v1, gamma 1/9    bandwidth changed, schema held
    D = v2, gamma 1/9    naive migration: both change together

Run:  uv run python scripts/experiment_role_schema.py [--json OUT]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel

import ts2net  # noqa: F401
from ts2net.networks.feature_schema import (
    REDUNDANT_V1_COLUMNS,
    V1_NAMES,
    V2_NAMES,
)

# The sibling characterization script owns the corpus and the raw-column
# reconstruction; reuse them so both analyses see identical inputs. Adding the
# script directory to sys.path keeps `uv run python scripts/...` working
# without callers having to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_role_features import NAMES as AUDIT_NAMES  # noqa: E402
from audit_role_features import corpus, raw_feature_matrix, zscore  # noqa: E402

warnings.filterwarnings("ignore")

# The audit script's short labels, in the same positional order as the schema.
SHORT_TO_SCHEMA = dict(zip(AUDIT_NAMES, V1_NAMES))
SCHEMA_TO_INDEX = {name: i for i, name in enumerate(V1_NAMES)}

GAMMA_POLICIES = {
    "historical_1_over_12": 1.0 / 12.0,
    "dimension_1_over_9": 1.0 / 9.0,
    "sklearn_spectral_default_1.0": 1.0,
}


def derive_redundant_columns(graphs) -> dict:
    """Find alias columns from the implementations, not from the audit's answer.

    A pair counts as an alias only if the two columns coincide on *every* graph
    in the corpus. That excludes coincidences: on a barbell, `core_number` and
    `triangles` happen to match, but they diverge elsewhere.
    """
    n_cols = len(V1_NAMES)
    always = {pair: True for pair in itertools.combinations(range(n_cols), 2)}
    for graph in graphs.values():
        _, raw = raw_feature_matrix(graph)
        std = zscore(raw)
        for pair in list(always):
            if not always[pair]:
                continue
            i, j = pair
            always[pair] = bool(np.allclose(std[:, i], std[:, j], atol=1e-9))
    return {
        V1_NAMES[j]: V1_NAMES[i] for (i, j), ok in always.items() if ok
    }


def schema_matrices(raw: np.ndarray) -> dict:
    """Standardized v1 and v2 matrices from the same raw columns."""
    v2_idx = [SCHEMA_TO_INDEX[name] for name in V2_NAMES]
    return {"v1": zscore(raw), "v2": zscore(raw[:, v2_idx])}


def pairwise(mat: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)
    return dist[np.triu_indices_from(dist, k=1)]


def unique_rows(mat: np.ndarray) -> int:
    return len(np.unique(np.round(mat, 9), axis=0))


def kmeans_labels(mat, n_roles, seed):
    return KMeans(n_clusters=n_roles, n_init=20, random_state=seed).fit_predict(mat)


def spectral_labels(mat, gamma, n_roles, seed):
    affinity = rbf_kernel(mat, gamma=gamma)
    return SpectralClustering(
        n_clusters=n_roles,
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
        n_init=20,
    ).fit_predict(affinity)


def agreement(a, b) -> dict:
    return {
        "ari": round(float(adjusted_rand_score(a, b)), 4),
        "nmi": round(float(normalized_mutual_info_score(a, b)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--n-roles", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[3363, 7, 42])
    args = ap.parse_args()

    graphs = corpus()
    report: dict = {"v1": list(V1_NAMES), "v2": list(V2_NAMES)}

    print("=" * 78)
    print("1. REDUNDANT COLUMNS DERIVED FROM THE IMPLEMENTATIONS")
    print("=" * 78)
    derived = derive_redundant_columns(graphs)
    declared = dict(REDUNDANT_V1_COLUMNS)
    print(f"  derived from {len(graphs)} graphs : {derived}")
    print(f"  declared in feature_schema  : {declared}")
    print(f"  agree: {derived == declared}")
    report["derived_redundant"] = derived
    report["declared_redundant"] = declared
    report["derived_matches_declared"] = derived == declared

    print()
    print("=" * 78)
    print("2. WELL-POSEDNESS  (identical feature rows make partitions arbitrary)")
    print("=" * 78)
    print(f"{'graph':<18} {'nodes':>6} {'unique v1 rows':>15} {'dup frac':>9}  class")
    posed = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = schema_matrices(raw)
        n = mats["v1"].shape[0]
        uniq = unique_rows(mats["v1"])
        dup = 1.0 - uniq / n
        ok = uniq >= args.n_roles and dup < 0.25
        posed[name] = ok
        print(
            f"{name:<18} {n:>6} {uniq:>15} {dup:>9.2f}  "
            f"{'well-posed' if ok else 'SYMMETRIC (reported separately)'}"
        )
    report["well_posed"] = posed

    print()
    print("=" * 78)
    print("3. KMEANS: schema effect (no kernel involved)")
    print("=" * 78)
    print(f"{'graph':<18} {'r(D1,D2)':>9} {'relL2':>7} {'ARI':>6} {'NMI':>6}")
    km_report = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = schema_matrices(raw)
        if np.allclose(mats["v1"], 0) and np.allclose(mats["v2"], 0):
            km_report[name] = "degenerate"
            print(f"{name:<18} {'both matrices all-zero':>9}")
            continue
        d1, d2 = pairwise(mats["v1"]), pairwise(mats["v2"])
        r = (
            float(np.corrcoef(d1, d2)[0, 1])
            if np.std(d1) > 1e-15 and np.std(d2) > 1e-15
            else float("nan")
        )
        n1, n2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
        rel = float(np.linalg.norm(n1 - n2) / np.linalg.norm(n1))
        rows = []
        for seed in args.seeds:
            rows.append(
                agreement(
                    kmeans_labels(mats["v1"], args.n_roles, seed),
                    kmeans_labels(mats["v2"], args.n_roles, seed),
                )
            )
        aris = [x["ari"] for x in rows]
        nmis = [x["nmi"] for x in rows]
        km_report[name] = {
            "pearson_r": round(r, 4),
            "rel_l2": round(rel, 4),
            "ari": aris,
            "nmi": nmis,
        }
        print(
            f"{name:<18} {r:>9.4f} {rel:>7.4f} "
            f"{np.median(aris):>6.3f} {np.median(nmis):>6.3f}"
        )
    report["kmeans"] = km_report

    print()
    print("=" * 78)
    print("4. SPECTRAL FACTORIAL:  A=v1@1/12  B=v2@1/12  C=v1@1/9  D=v2@1/9")
    print("=" * 78)
    print(
        f"{'graph':<18} {'A~B':>6} {'A~C':>6} {'A~D':>6} {'C~D':>6} {'B~D':>6}  "
        "(ARI; A~B schema only, A~C gamma only, A~D naive migration)"
    )
    cells = {
        "A": ("v1", 1 / 12),
        "B": ("v2", 1 / 12),
        "C": ("v1", 1 / 9),
        "D": ("v2", 1 / 9),
    }
    sp_report = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = schema_matrices(raw)
        if mats["v1"].shape[0] <= args.n_roles or np.allclose(mats["v1"], 0):
            sp_report[name] = "skipped"
            continue
        per_seed = {k: [] for k in ("A~B", "A~C", "A~D", "C~D", "B~D")}
        for seed in args.seeds:
            lab = {
                key: spectral_labels(mats[schema], gamma, args.n_roles, seed)
                for key, (schema, gamma) in cells.items()
            }
            for key in per_seed:
                left, right = key.split("~")
                per_seed[key].append(adjusted_rand_score(lab[left], lab[right]))
        med = {k: float(np.median(v)) for k, v in per_seed.items()}
        sp_report[name] = {k: [round(x, 4) for x in v] for k, v in per_seed.items()}
        print(
            f"{name:<18} {med['A~B']:>6.3f} {med['A~C']:>6.3f} {med['A~D']:>6.3f} "
            f"{med['C~D']:>6.3f} {med['B~D']:>6.3f}"
        )
    report["spectral_factorial"] = sp_report

    print()
    print("=" * 78)
    print("5. GAMMA POLICIES FOR v2  (agreement with production A = v1@1/12)")
    print("=" * 78)
    print(f"{'graph':<18} " + " ".join(f"{k:>28}" for k in GAMMA_POLICIES))
    gp_report = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = schema_matrices(raw)
        if mats["v1"].shape[0] <= args.n_roles or np.allclose(mats["v1"], 0):
            continue
        row = {}
        for policy, gamma in GAMMA_POLICIES.items():
            vals = [
                adjusted_rand_score(
                    spectral_labels(mats["v1"], 1 / 12, args.n_roles, seed),
                    spectral_labels(mats["v2"], gamma, args.n_roles, seed),
                )
                for seed in args.seeds
            ]
            row[policy] = round(float(np.median(vals)), 4)
        gp_report[name] = row
        print(f"{name:<18} " + " ".join(f"{row[k]:>28.3f}" for k in GAMMA_POLICIES))
    report["gamma_policies"] = gp_report

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
