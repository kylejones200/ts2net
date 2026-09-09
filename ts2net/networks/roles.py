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
else:
    _tri_rs = _rs.triangles_per_node


#: Default RBF kernel bandwidth for :func:`node_roles_spectral`.
#:
#: This is a named policy, deliberately independent of the feature matrix'
#: width. It previously read ``1 / X.shape[1]``, which made the number of
#: feature columns a hyperparameter of the clustering algorithm: adding or
#: removing a column silently retuned the kernel. See
#: ``docs/audits/role_schema_v2_design.md``, whose factorial experiment shows
#: the schema effect and the bandwidth effect interact by up to 0.882 ARI per
#: graph, so a change to one must not drag the other along.
#:
#: The value is the historical ``1/12`` that the twelve-column schema produced,
#: pinned so that removing the three redundant columns changes the feature
#: geometry and nothing else. It is not derived from the current column count
#: and must not be.
#:
#: Pass ``gamma`` explicitly to override. A data-driven bandwidth such as the
#: median heuristic is a reasonable future default, but it is a separate
#: decision with its own evidence.
DEFAULT_SPECTRAL_GAMMA = 1.0 / 12.0


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


def _motif_features(G: nx.Graph, nodes: List) -> np.ndarray:
    H = G.to_undirected()
    tri = _triangles_per_node(H)
    deg = np.array([H.degree(u) for u in nodes], dtype=np.int64)
    wedges = np.maximum(deg * (deg - 1) // 2 - tri, 0)
    return np.vstack([tri, wedges]).T.astype(float)


def role_features_extended(G: nx.Graph) -> Tuple[List, np.ndarray]:
    """Structural role features per node, standardized.

    Returns ``(nodes, X)`` where ``X`` has one row per node and one column per
    entry of :data:`ts2net.networks.feature_schema.ROLE_FEATURES_V2`, in that
    order. Import the schema to address a column by meaning rather than by
    position.

    The columns are nine independent signals. Three further columns shipped
    before this release -- ``ego_edges``, ``ego_density`` and ``core_score`` --
    were exact aliases of ``triangles``, ``clustering`` and ``core_number``
    respectively, and are removed; see
    ``docs/audits/role_schema_v2_design.md``. ``wedges`` is retained although
    it is determined by ``degree`` and ``triangles`` as
    ``C(degree, 2) - triangles``, because it is quadratic in degree and so
    spans a direction neither parent does.
    """
    H = G.to_undirected()
    nodes, Xbasic = _role_features_basic(H)
    nodes = list(nodes)
    tri_wedge = _motif_features(H, nodes)
    X = np.hstack([Xbasic, tri_wedge])
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
    """Cluster nodes into roles by spectral clustering of the feature matrix.

    ``gamma`` is the RBF kernel bandwidth. When omitted,
    :data:`DEFAULT_SPECTRAL_GAMMA` is used -- a fixed, named value that does
    not depend on how many feature columns arrive.
    """
    from sklearn.cluster import SpectralClustering

    nodes, X = role_features_extended(G)
    if affinity == "rbf":
        from sklearn.metrics.pairwise import rbf_kernel

        if gamma is None:
            gamma = DEFAULT_SPECTRAL_GAMMA
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
