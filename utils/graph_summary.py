def graph_summary(
    G: Union[nx.Graph, nx.DiGraph],
    motifs: str | None = None,
    motif_samples: int | None = None,
    seed: int = 3363,
) -> dict:
    und = G.to_undirected() if G.is_directed() else G
    deg = dict(und.degree())
    deg_vals = np.array(list(deg.values()), dtype=float)
    out = {
        "n": und.number_of_nodes(),
        "m": und.number_of_edges(),
        "deg_mean": float(deg_vals.mean()) if deg_vals.size else 0.0,
        "deg_std": float(deg_vals.std(ddof=1)) if deg_vals.size > 1 else 0.0,
        "assortativity": (
            nx.degree_assortativity_coefficient(und)
            if und.number_of_edges()
            else np.nan
        ),
        "avg_clustering": (
            nx.average_clustering(und) if und.number_of_nodes() else np.nan
        ),
    }
    out.update(small_world_summary(und))
    out.update(motif_summary(und))

    # extra motifs on demand
    if motifs in ("directed3", "all") and G.is_directed():
        out["motifs_directed_3"] = directed_3node_motifs(
            G, max_samples=motif_samples, seed=seed
        )
    if motifs in ("undirected4", "all"):
        base = G.to_undirected() if G.is_directed() else G
        out["motifs_undirected_4"] = undirected_4node_motifs(
            base, max_samples=motif_samples, seed=seed
        )
    return out


# --- directed 3-node motifs (connected) ---
def directed_3node_motifs(
    G: nx.DiGraph, max_samples: int | None = None, seed: int = 3363
) -> dict:
    if not G.is_directed():
        raise ValueError("Graph must be directed.")
    nodes = list(G.nodes())
    rng = random.Random(seed)
    counts = {}
    total = 0
    for trio in _sampled_combinations(nodes, 3, max_samples, rng):
        sub = G.subgraph(trio).copy()
        if not nx.is_weakly_connected(sub):
            continue
        # relabel to 0..2
        mapping = dict(zip(sub.nodes(), range(3)))
        sub = nx.relabel_nodes(sub, mapping)
        # 6-bit code for ordered pairs (0->1,0->2,1->0,1->2,2->0,2->1)
        bits = [
            int(sub.has_edge(0, 1)),
            int(sub.has_edge(0, 2)),
            int(sub.has_edge(1, 0)),
            int(sub.has_edge(1, 2)),
            int(sub.has_edge(2, 0)),
            int(sub.has_edge(2, 1)),
        ]
        code = "".join(map(str, bits))
        counts[code] = counts.get(code, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: {"count": v, "freq": v / total} for k, v in counts.items()}


# --- undirected 4-node motifs (connected types) ---
def _deg_seq(sub: nx.Graph) -> list[int]:
    return sorted([d for _, d in sub.degree()])


def undirected_4node_motifs(
    G: nx.Graph, max_samples: int | None = None, seed: int = 3363
) -> dict:
    if G.is_directed():
        raise ValueError("Graph must be undirected.")
    nodes = list(G.nodes())
    rng = random.Random(seed)
    out = {"K4": 0, "C4": 0, "triangle_tail": 0, "P4": 0, "K1,3": 0, "other": 0}
    total = 0
    for quad in _sampled_combinations(nodes, 4, max_samples, rng):
        sub = G.subgraph(quad)
        if not nx.is_connected(sub):
            continue
        m = sub.number_of_edges()
        d = _deg_seq(sub)
        if m == 6:
            out["K4"] += 1
        elif m == 4 and d == [2, 2, 2, 2]:
            out["C4"] += 1
        elif m == 4 and d == [1, 2, 2, 3]:
            out["triangle_tail"] += 1
        elif m == 3 and d == [1, 1, 2, 2]:
            out["P4"] += 1
        elif m == 3 and d == [1, 1, 1, 3]:
            out["K1,3"] += 1
        else:
            out["other"] += 1
        total += 1
    if total == 0:
        return {}
    return {k: {"count": v, "freq": v / total} for k, v in out.items()}


# --- existing small motifs (keep as-is) ---
def triangle_count(G: nx.Graph) -> int:
    H = G.to_undirected() if G.is_directed() else G
    tri_dict = nx.triangles(H)
    return int(sum(tri_dict.values()) // 3)


def wedge_count(G: nx.Graph) -> int:
    H = G.to_undirected() if G.is_directed() else G
    deg = np.array([d for _, d in H.degree()], dtype=np.int64)
    return int(np.sum(deg * (deg - 1) // 2))


def motif_summary(G: Union[nx.Graph, nx.DiGraph]) -> dict:
    und = G.to_undirected() if G.is_directed() else G
    T = triangle_count(und)
    W = wedge_count(und)
    n = und.number_of_nodes()
    N3 = n * (n - 1) * (n - 2) // 6
    S = max(N3 - (T + W), 0)
    return {"triangles": T, "wedges": W, "other_3_node": S, "N3": N3}
