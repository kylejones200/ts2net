from itertools import combinations


def directed_3node_motifs(G: nx.DiGraph) -> dict:
    """Count connected directed 3-node motifs using adjacency pattern codes."""
    if not G.is_directed():
        raise ValueError("Graph must be directed.")
    motifs = {}
    nodes = list(G.nodes())
    for trio in combinations(nodes, 3):
        sub = G.subgraph(trio).copy()
        # relabel to 0,1,2 for canonical form
        mapping = dict(zip(sub.nodes(), range(3)))
        sub = nx.relabel_nodes(sub, mapping)
        # encode adjacency as bitstring of length 6 (excluding self-loops)
        edges = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.append(1 if sub.has_edge(i, j) else 0)
        code = "".join(map(str, edges))
        motifs[code] = motifs.get(code, 0) + 1
    total = sum(motifs.values())
    return {k: {"count": v, "freq": v / total} for k, v in motifs.items()}


def undirected_4node_motifs(G: nx.Graph) -> dict:
    """Count simple undirected 4-node motifs by isomorphism type."""
    if G.is_directed():
        raise ValueError("Graph must be undirected.")
    motifs = {
        "4-clique": 0,
        "square": 0,
        "triangle-tail": 0,
        "4-chain": 0,
        "4-star": 0,
        "other": 0,
    }
    for quad in combinations(G.nodes(), 4):
        sub = G.subgraph(quad)
        m = sub.number_of_edges()
        if m == 6:
            motifs["4-clique"] += 1
        elif m == 4 and nx.cycle_graph(4).edges() <= set(map(frozenset, sub.edges())):
            motifs["square"] += 1
        elif m == 4 and any(
            len(list(sub.subgraph(c).edges())) == 3
            for c in combinations(sub.nodes(), 3)
        ):
            motifs["triangle-tail"] += 1
        elif m == 3 and nx.is_connected(sub):
            motifs["4-chain"] += 1
        elif m == 3 and not nx.is_connected(sub):
            motifs["4-star"] += 1
        else:
            motifs["other"] += 1
    total = sum(motifs.values())
    return {
        k: {"count": v, "freq": v / total if total else 0} for k, v in motifs.items()
    }
