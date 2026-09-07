from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Literal, List, Optional

from ._standardize import standardize
from .communities import _role_features_basic

# The Rust fast path. Only the extension being absent is a reason to fall back
# to networkx: a missing symbol means the extension was built from the wrong
# source or the binding was dropped, which is a defect that must surface rather
# than silently degrade into a slower code path. So the module import is
# guarded and the attribute lookups are not.
try:
    import ts2net_rs as _rs
except ImportError:  # pragma: no cover - exercised only without the extension
    _rs = None

if _rs is None:
    _tri_rs = None
    _ego_rs = None
    _core_rs = None
else:
    _tri_rs = _rs.triangles_per_node
    _ego_rs = _rs.ego_edge_counts
    _core_rs = _rs.core_numbers


def _edges_array(G: nx.Graph) -> Tuple[int, np.ndarray, bool, List, Dict]:
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    undirected = not G.is_directed()
    E = np.empty((G.number_of_edges(), 2), dtype=np.uint64)
    k = 0
    for u, v in G.edges():
        E[k, 0] = idx[u]
        E[k, 1] = idx[v]
        k += 1
    return len(nodes), E, undirected, nodes, idx


def _triangles_per_node(G: nx.Graph) -> np.ndarray:
    n, E, _, _, _ = _edges_array(G)
    if _tri_rs is not None:
        return np.array(_tri_rs(n, E), dtype=np.int64)
    # fallback
    H = G.to_undirected()
    tri = nx.triangles(H)
    return np.array([tri[u] for u in H.nodes()], dtype=np.int64)


def _ego_edges_per_node(G: nx.Graph) -> np.ndarray:
    n, E, _, nodes, _ = _edges_array(G)
    if _ego_rs is not None:
        return np.array(_ego_rs(n, E), dtype=np.int64)
    out = np.zeros(len(nodes), dtype=np.int64)
    H = G.to_undirected()
    for i, u in enumerate(nodes):
        nbrs = list(H.neighbors(u))
        sub = H.subgraph(nbrs)
        out[i] = sub.number_of_edges()
    return out


def _core_number(G: nx.Graph) -> np.ndarray:
    n, E, _, nodes, _ = _edges_array(G)
    if _core_rs is not None:
        return np.array(_core_rs(n, E), dtype=np.int64)
    core = nx.core_number(G.to_undirected())
    return np.array([core[u] for u in nodes], dtype=np.int64)


def _egonet_density(G: nx.Graph, nodes: List) -> np.ndarray:
    H = G.to_undirected()
    out = np.zeros(len(nodes), float)
    for i, u in enumerate(nodes):
        nbrs = list(H.neighbors(u))
        k = len(nbrs)
        if k <= 1:
            out[i] = 0.0
            continue
        m = H.subgraph(nbrs).number_of_edges()
        out[i] = 2.0 * m / (k * (k - 1))
    return out


def _motif_features(G: nx.Graph, nodes: List) -> np.ndarray:
    H = G.to_undirected()
    tri = _triangles_per_node(H)
    deg = np.array([H.degree(u) for u in nodes], dtype=np.int64)
    wedges = np.maximum(deg * (deg - 1) // 2 - tri, 0)
    return np.vstack([tri, wedges]).T.astype(float)


def _core_periphery_scores(G: nx.Graph, nodes: List) -> np.ndarray:
    H = G.to_undirected()
    c = _core_number(H).astype(float)
    if c.max() > 0:
        c = c / c.max()
    return c


def role_features_extended(G: nx.Graph) -> Tuple[List, np.ndarray]:
    H = G.to_undirected()
    nodes, Xbasic = _role_features_basic(H)
    nodes = list(nodes)
    tri_wedge = _motif_features(H, nodes)
    ego_edges = _ego_edges_per_node(H).astype(float).reshape(-1, 1)
    ego_density = _egonet_density(H, nodes).reshape(-1, 1)
    core_score = _core_periphery_scores(H, nodes).reshape(-1, 1)
    X = np.hstack([Xbasic, tri_wedge, ego_edges, ego_density, core_score])
    # Single standardization boundary for the whole matrix; the columns above
    # are raw. See ts2net.networks._standardize.
    X = standardize(X)
    return nodes, X


def node_roles_kmeans(G: nx.Graph, n_roles: int = 6, seed: int = 3363) -> Dict:
    from sklearn.cluster import KMeans

    nodes, X = role_features_extended(G)
    km = KMeans(n_clusters=int(n_roles), n_init=20, random_state=seed)
    lab = km.fit_predict(X)
    return {n: int(r) for n, r in zip(nodes, lab)}


def node_roles_spectral(
    G: nx.Graph,
    n_roles: int = 6,
    seed: int = 3363,
    affinity: Literal["rbf", "cosine"] = "rbf",
    gamma: Optional[float] = None,
) -> Dict:
    from sklearn.cluster import SpectralClustering

    nodes, X = role_features_extended(G)
    if affinity == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        if gamma is None:
            gamma = 1.0 / X.shape[1]
        A = rbf_kernel(X, gamma=float(gamma))
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        A = cosine_similarity(X)
    sc = SpectralClustering(
        n_clusters=int(n_roles),
        affinity="precomputed",
        random_state=seed,
        assign_labels="kmeans",
        n_init=20,
    )
    lab = sc.fit_predict(A)
    return {n: int(r) for n, r in zip(nodes, lab)}
