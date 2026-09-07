#!/usr/bin/env python3
"""Characterization of the feature matrix returned by `role_features_extended`.

Analysis only. This script imports production code and never modifies it; the
de-duplicated and triplicated comparison matrices below are built here, not in
`ts2net.networks.roles`.

`role_features_extended` returns twelve columns. This measures whether they
carry twelve distinct signals, and what the redundancy does to the geometry
that KMeans and SpectralClustering see.

Run:  uv run python scripts/audit_role_features.py [--json OUT]
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings

import networkx as nx
import numpy as np

import ts2net  # noqa: F401  -- side effect: makes the extension importable
from ts2net.networks import roles

warnings.filterwarnings("ignore")

# Column order produced by role_features_extended: seven from
# _role_features_basic, then tri, wedges, ego_edges, ego_density, core_score.
NAMES = [
    "deg", "cc", "pr", "ev", "core", "btw", "clo",
    "tri", "wedges", "ego_edges", "ego_density", "core_score",
]

# The three signals that are duplicated, and the copy this audit treats as
# canonical. Used only to build comparison matrices.
ALIASES = {"ego_edges": "tri", "ego_density": "cc", "core_score": "core"}
CANONICAL = [n for n in NAMES if n not in ALIASES]


def zscore(x: np.ndarray) -> np.ndarray:
    """The documented standardization rule, implemented independently here.

    Mirrors ts2net.networks._standardize.standardize without importing it, so
    the reconstruction check below compares two implementations rather than one
    implementation against itself: degenerate columns become exactly zero, and
    every other column is centred and scaled to unit variance.
    """
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    if x.shape[0] < 2:
        return out
    spread = x.std(axis=0, ddof=1)
    scale = np.maximum(np.abs(x.mean(axis=0)), np.abs(x).max(axis=0))
    keep = spread > np.maximum(1e-12, 1e-12 * scale)
    if keep.any():
        sub = x[:, keep]
        out[:, keep] = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=1)
    return out


def raw_feature_matrix(graph: nx.Graph):
    """The twelve columns *before* any standardization.

    Mirrors _role_features_basic and the roles.py helpers exactly. Production
    standardizes inside _role_features_basic and again in
    role_features_extended, so the raw values are not otherwise observable.
    """
    und = graph.to_undirected()
    nodes = list(und.nodes())
    deg = np.array([und.degree(n) for n in nodes], float)
    cc = np.array(list(nx.clustering(und).values()), float)
    pr = np.array(list(nx.pagerank(und).values()), float)
    try:
        ev_d = nx.eigenvector_centrality_numpy(und)
        ev = np.array([ev_d[n] for n in nodes], float)
    except Exception:
        ev = np.zeros_like(deg)
    core_d = nx.core_number(und)
    core = np.array([core_d[n] for n in nodes], float)
    btw = np.array(
        list(nx.betweenness_centrality(und, normalized=True).values()), float
    )
    clo = np.array(list(nx.closeness_centrality(und).values()), float)

    tri_wedge = roles._motif_features(und, nodes)
    tri, wedges = tri_wedge[:, 0], tri_wedge[:, 1]
    ego_edges = roles._ego_edges_per_node(und).astype(float)
    ego_density = roles._egonet_density(und, nodes)
    core_score = roles._core_periphery_scores(und, nodes).astype(float)

    raw = np.vstack([
        deg, cc, pr, ev, core, btw, clo,
        tri, wedges, ego_edges, ego_density, core_score,
    ]).T
    return nodes, raw


def validate_reconstruction(graph: nx.Graph) -> float:
    """Max abs difference between z(raw) and what production returns."""
    _, raw = raw_feature_matrix(graph)
    _, prod = roles.role_features_extended(graph)
    return float(np.max(np.abs(zscore(raw) - prod)))


def determinism(graph: nx.Graph, repeats: int = 5) -> float:
    """Max spread across repeated identical calls to role_features_extended.

    nx.eigenvector_centrality_numpy runs ARPACK from a random start vector. On
    a graph whose eigenvector centrality is uniform (any vertex-transitive
    graph) the result is uniform only to ~1e-15, and the sign/pattern of that
    float noise changes between calls.
    """
    runs = [roles.role_features_extended(graph)[1] for _ in range(repeats)]
    return float(max(np.max(np.abs(runs[0] - r)) for r in runs[1:]))


def noise_amplification(graph: nx.Graph) -> list:
    """Columns that are numerically constant raw but unit-variance after the pipeline.

    _role_features_basic z-scores its seven columns, and role_features_extended
    z-scores the concatenation again. The first z-score's `+ 1e-12` guard keeps
    a near-constant column small but non-zero; the second sees a std far above
    the guard and renormalises that float noise to unit variance.
    """
    _, raw = raw_feature_matrix(graph)
    _, prod = roles.role_features_extended(graph)
    out = []
    for i, name in enumerate(NAMES):
        raw_std = float(np.std(raw[:, i], ddof=1))
        scale = max(abs(float(np.mean(raw[:, i]))), 1.0)
        out_std = float(np.std(prod[:, i], ddof=1))
        if raw_std / scale < 1e-10 and out_std > 0.5:
            out.append((name, raw_std, out_std))
    return out


def corpus():
    """Graph families chosen to expose different structural regimes."""
    g = {
        "path_20": nx.path_graph(20),
        "cycle_20": nx.cycle_graph(20),
        "star_19": nx.star_graph(19),
        "wheel_20": nx.wheel_graph(20),
        "complete_12": nx.complete_graph(12),
        "bipartite_5_7": nx.complete_bipartite_graph(5, 7),
        "tree_2_4": nx.balanced_tree(2, 4),
        "grid_5x5": nx.grid_2d_graph(5, 5),
        "barbell_8_3": nx.barbell_graph(8, 3),
        "karate": nx.karate_club_graph(),
        "les_mis": nx.les_miserables_graph(),
    }
    for seed in (1, 2, 3):
        g[f"er_40_p15_s{seed}"] = nx.gnp_random_graph(40, 0.15, seed=seed)
        g[f"ws_40_k4_p1_s{seed}"] = nx.watts_strogatz_graph(40, 4, 0.1, seed=seed)
        g[f"ba_40_m3_s{seed}"] = nx.barabasi_albert_graph(40, 3, seed=seed)
    return g


# --------------------------------------------------------------------------
# 1. algebraic identities
# --------------------------------------------------------------------------

def check_identities(graph: nx.Graph) -> dict:
    und = graph.to_undirected()
    nodes = list(und.nodes())
    deg = np.array([und.degree(n) for n in nodes], float)
    tri = roles._triangles_per_node(und).astype(float)
    ego = roles._ego_edges_per_node(und).astype(float)
    dens = roles._egonet_density(und, nodes)
    cc = np.array(list(nx.clustering(und).values()), float)

    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = np.where(deg >= 2, 2.0 * tri / (deg * (deg - 1)), 0.0)

    low = deg <= 1
    return {
        "ego_equals_tri": bool(np.array_equal(ego, tri)),
        "density_equals_formula": bool(np.allclose(dens, predicted, atol=1e-12)),
        "density_equals_clustering": bool(np.allclose(dens, cc, atol=1e-12)),
        "n_deg_le_1": int(low.sum()),
        "deg_le_1_density_all_zero": (
            bool(np.all(dens[low] == 0.0)) if low.any() else None
        ),
        "deg_le_1_tri_all_zero": (
            bool(np.all(tri[low] == 0.0)) if low.any() else None
        ),
    }


# --------------------------------------------------------------------------
# 2. column structure
# --------------------------------------------------------------------------

def affine_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R^2 of the best y = a*x + b fit. 1.0 means a deterministic affine map."""
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def column_structure(raw: np.ndarray) -> dict:
    n_cols = raw.shape[1]
    zero_var = [NAMES[i] for i in range(n_cols) if np.std(raw[:, i], ddof=1) < 1e-12]
    z = zscore(raw)
    exact, affine = [], []
    for i, j in itertools.combinations(range(n_cols), 2):
        if np.allclose(raw[:, i], raw[:, j], atol=1e-12):
            exact.append((NAMES[i], NAMES[j], "raw"))
        elif np.allclose(z[:, i], z[:, j], atol=1e-9):
            exact.append((NAMES[i], NAMES[j], "after z-score"))
        else:
            r2 = affine_r2(raw[:, i], raw[:, j])
            if not np.isnan(r2) and r2 > 1 - 1e-9:
                affine.append((NAMES[i], NAMES[j], round(float(r2), 12)))
    return {
        "zero_variance": zero_var,
        "exact_duplicates": exact,
        "affine_pairs": affine,
        "rank_raw": int(np.linalg.matrix_rank(raw, tol=1e-9)),
        "rank_z": int(np.linalg.matrix_rank(z, tol=1e-9)),
        "n_cols": n_cols,
    }


def degree_dependence(raw: np.ndarray) -> dict:
    deg = raw[:, 0]
    out = {}
    for i, name in enumerate(NAMES):
        if name == "deg":
            continue
        out[name] = {
            "pearson_vs_deg": None
            if np.std(raw[:, i]) < 1e-15 or np.std(deg) < 1e-15
            else round(float(np.corrcoef(deg, raw[:, i])[0, 1]), 4),
            "spearman_vs_deg": None
            if np.isnan(spearman(deg, raw[:, i]))
            else round(spearman(deg, raw[:, i]), 4),
        }
    return out


# --------------------------------------------------------------------------
# 3. downstream geometry and clustering
# --------------------------------------------------------------------------

def comparison_matrices(raw: np.ndarray):
    """z-scored full (12), de-duplicated (9), and triangle-triplicated (13)."""
    idx = {n: i for i, n in enumerate(NAMES)}
    full = zscore(raw)
    canon = zscore(raw[:, [idx[n] for n in CANONICAL]])
    trip = zscore(np.hstack([raw, raw[:, [idx["tri"]]]]))  # tri appears 3x
    return {"full12": full, "canonical9": canon, "triplicated13": trip}


def pairwise(mat: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)
    return d[np.triu_indices_from(d, k=1)]


def geometry_effect(mats: dict) -> dict:
    """How much the redundancy changes distances before any clustering."""
    out = {}
    base = pairwise(mats["full12"])
    for name in ("canonical9", "triplicated13"):
        other = pairwise(mats[name])
        # Scale-free comparison: dimensionality differs, so normalise each
        # distance vector before comparing shape.
        bn, on = base / np.linalg.norm(base), other / np.linalg.norm(other)
        out[name] = {
            "pearson_r_of_distances": round(float(np.corrcoef(base, other)[0, 1]), 6),
            "relative_l2_diff_normalised": round(
                float(np.linalg.norm(bn - on) / np.linalg.norm(bn)), 6
            ),
        }
    return out


def cluster_effect(mats: dict, n_roles: int, seed: int) -> dict:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.pairwise import rbf_kernel

    def km(mat):
        return KMeans(n_clusters=n_roles, n_init=20, random_state=seed).fit_predict(mat)

    def sc(mat, gamma):
        affinity = rbf_kernel(mat, gamma=gamma)
        return SpectralClustering(
            n_clusters=n_roles, affinity="precomputed", random_state=seed,
            assign_labels="kmeans", n_init=20,
        ).fit_predict(affinity)

    base = mats["full12"]
    res = {}
    lab_km_base = km(base)
    # Production picks gamma = 1/n_features, which itself changes with column
    # count. Hold it fixed at the 12-column value to isolate the duplication.
    gamma_fixed = 1.0 / base.shape[1]
    lab_sc_base = sc(base, gamma_fixed)

    for name in ("canonical9", "triplicated13"):
        mat = mats[name]
        res[name] = {
            "kmeans_ari": round(float(adjusted_rand_score(lab_km_base, km(mat))), 4),
            "kmeans_nmi": round(
                float(normalized_mutual_info_score(lab_km_base, km(mat))), 4
            ),
            "spectral_ari_fixed_gamma": round(
                float(adjusted_rand_score(lab_sc_base, sc(mat, gamma_fixed))), 4
            ),
            "spectral_ari_production_gamma": round(
                float(adjusted_rand_score(lab_sc_base, sc(mat, 1.0 / mat.shape[1]))), 4
            ),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--n-roles", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[3363, 7, 42])
    args = ap.parse_args()

    graphs = corpus()
    report: dict = {"graphs": {}, "names": NAMES, "canonical": CANONICAL}

    print("=" * 78)
    print("0a. DETERMINISM  (repeat calls on an unchanged graph)")
    print("=" * 78)
    det = {}
    nondet = []
    for name, graph in graphs.items():
        d = determinism(graph)
        det[name] = d
        if d > 1e-9:
            nondet.append(name)
            print(f"  {name:<18} NON-DETERMINISTIC  max spread {d:.3e}")
    print(f"  {len(graphs) - len(nondet)}/{len(graphs)} graphs reproducible to 1e-9")
    report["determinism"] = det
    report["nondeterministic_graphs"] = nondet

    print()
    print("=" * 78)
    print("0b. NOISE AMPLIFICATION  (numerically constant raw -> unit variance out)")
    print("=" * 78)
    amp = {}
    for name, graph in graphs.items():
        a = noise_amplification(graph)
        if a:
            amp[name] = [(c, rs, os_) for c, rs, os_ in a]
            cols = ", ".join(
                f"{c} (raw std {rs:.1e} -> out std {os_:.2f})" for c, rs, os_ in a
            )
            print(f"  {name:<18} {cols}")
    if not amp:
        print("  none")
    report["noise_amplification"] = amp

    print()
    print("=" * 78)
    print("0c. RECONSTRUCTION CHECK  (deterministic graphs only)")
    print("=" * 78)
    worst = 0.0
    for name, graph in graphs.items():
        if name in nondet:
            continue
        worst = max(worst, validate_reconstruction(graph))
    print(f"max abs difference over {len(graphs) - len(nondet)} graphs: {worst:.3e}")
    report["reconstruction_max_abs_diff"] = worst

    print()
    print("=" * 78)
    print("1. ALGEBRAIC IDENTITIES")
    print("=" * 78)
    print(
        f"{'graph':<18} {'ego==tri':>9} {'dens==2T/k(k-1)':>17} "
        f"{'dens==cc':>9} {'deg<=1':>7}"
    )
    ident_all = {}
    for name, graph in graphs.items():
        r = check_identities(graph)
        ident_all[name] = r
        print(
            f"{name:<18} {str(r['ego_equals_tri']):>9} "
            f"{str(r['density_equals_formula']):>17} "
            f"{str(r['density_equals_clustering']):>9} {r['n_deg_le_1']:>7}"
        )
    report["identities"] = ident_all

    print()
    print("=" * 78)
    print("2. COLUMN STRUCTURE  (raw, pre-standardization)")
    print("=" * 78)
    print(f"{'graph':<18} {'rank':>5} {'zero-var columns':<28} duplicates")
    struct_all = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        s = column_structure(raw)
        struct_all[name] = s
        dup = ", ".join(f"{a}={b}" for a, b, _ in s["exact_duplicates"])
        zv = ", ".join(s["zero_variance"]) or "-"
        print(f"{name:<18} {s['rank_raw']:>2}/12 {zv:<28} {dup}")
    report["structure"] = struct_all

    print()
    print("=" * 78)
    print("3. GEOMETRY: does the redundancy move node distances?")
    print("=" * 78)
    print(
        f"{'graph':<18} {'r(D12,D9)':>10} {'relL2 9':>9} "
        f"{'r(D12,D13)':>11} {'relL2 13':>9}"
    )
    geo_all = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = comparison_matrices(raw)
        if np.allclose(mats["full12"], 0):
            print(f"{name:<18} {'degenerate (all columns constant)':>10}")
            geo_all[name] = "degenerate"
            continue
        g = geometry_effect(mats)
        geo_all[name] = g
        print(
            f"{name:<18} {g['canonical9']['pearson_r_of_distances']:>10.4f} "
            f"{g['canonical9']['relative_l2_diff_normalised']:>9.4f} "
            f"{g['triplicated13']['pearson_r_of_distances']:>11.4f} "
            f"{g['triplicated13']['relative_l2_diff_normalised']:>9.4f}"
        )
    report["geometry"] = geo_all

    print()
    print("=" * 78)
    print("4. CLUSTERING: 12-column result vs de-duplicated and triplicated")
    print("=" * 78)
    print(
        f"{'graph':<18} {'seed':>5} {'kmARI9':>7} {'kmNMI9':>7} "
        f"{'scARI9':>7} {'kmARI13':>8} {'scARI13':>8}"
    )
    clus_all = {}
    for name, graph in graphs.items():
        _, raw = raw_feature_matrix(graph)
        mats = comparison_matrices(raw)
        if np.allclose(mats["full12"], 0) or mats["full12"].shape[0] <= args.n_roles:
            clus_all[name] = "skipped (degenerate or too few nodes)"
            continue
        clus_all[name] = {}
        for seed in args.seeds:
            c = cluster_effect(mats, args.n_roles, seed)
            clus_all[name][seed] = c
            print(
                f"{name:<18} {seed:>5} "
                f"{c['canonical9']['kmeans_ari']:>7.3f} "
                f"{c['canonical9']['kmeans_nmi']:>7.3f} "
                f"{c['canonical9']['spectral_ari_fixed_gamma']:>7.3f} "
                f"{c['triplicated13']['kmeans_ari']:>8.3f} "
                f"{c['triplicated13']['spectral_ari_fixed_gamma']:>8.3f}"
            )
    report["clustering"] = clus_all

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
