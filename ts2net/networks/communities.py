from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Literal, Optional

try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = None
    la = None


def _nx_to_igraph(G: nx.Graph) -> ig.Graph:
    if ig is None:
        raise RuntimeError("Install python-igraph and leidenalg.")
    mapping = {n: i for i, n in enumerate(G.nodes())}
    H = ig.Graph()
    H.add_vertices(len(mapping))
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    H.add_edges(edges)
    return H, mapping


def _community_labels_from_sets(comms) -> Dict:
    """Convert community sets to node->community_id mapping."""
    return {u: cid for cid, C in enumerate(comms) for u in C}


def detect_communities(
    G: nx.Graph,
    method: Literal["louvain", "leiden", "label_propagation", "greedy"] = "louvain",
    weight: Optional[str] = "weight",
    seed: Optional[int] = 3363,
) -> Dict:
    H = G.to_undirected()
    
    def _louvain():
        try:
            import community as community_louvain
        except Exception:
            raise RuntimeError("Install python-louvain.")
        return community_louvain.best_partition(H, weight=weight, random_state=seed)
    
    def _label_propagation():
        comms = nx.algorithms.community.asyn_lpa_communities(H, weight=weight, seed=seed)
        return _community_labels_from_sets(comms)
    
    def _greedy():
        comms = nx.algorithms.community.greedy_modularity_communities(H, weight=weight)
        return _community_labels_from_sets(comms)
    
    def _leiden():
        if ig is None or la is None:
            raise RuntimeError("Install python-igraph and leidenalg.")
        Gi, mapping = _nx_to_igraph(H)
        w = None
        if weight is not None:
            w = [H[u][v].get(weight, 1.0) for u, v in H.edges()]
            Gi.es["w"] = w
        part = la.find_partition(
            Gi, la.RBConfigurationVertexPartition, weights="w" if w else None, seed=seed
        )
        inv = {i: n for n, i in mapping.items()}
        return {inv[i]: cid for cid, C in enumerate(part) for i in C}
    
    method_handlers = {
        "louvain": _louvain,
        "label_propagation": _label_propagation,
        "greedy": _greedy,
        "leiden": _leiden,
    }
    
    handler = method_handlers.get(method)
    if handler is None:
        raise ValueError(f"Unknown method: {method}")
    
    return handler()


def _role_features_basic(G: nx.Graph) -> np.ndarray:
    H = G.to_undirected()
    nodes = list(H.nodes())
    deg = np.array([H.degree(n) for n in nodes], float)
    cc = np.array(list(nx.clustering(H).values()), float)
    pr = np.array(list(nx.pagerank(H).values()), float)
    try:
        ev = nx.eigenvector_centrality_numpy(H)
        ev = np.array([ev[n] for n in nodes], float)
    except Exception:
        ev = np.zeros_like(deg)
    core = nx.core_number(H)
    core = np.array([core[n] for n in nodes], float)
    btw = np.array(list(nx.betweenness_centrality(H, normalized=True).values()), float)
    clo = np.array(list(nx.closeness_centrality(H).values()), float)
    X = np.vstack([deg, cc, pr, ev, core, btw, clo]).T
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-12)
    return nodes, X


def node_roles(
    G: nx.Graph,
    n_roles: int = 4,
    features: Literal["basic"] = "basic",
    n_init: int = 10,
    seed: int = 3363,
) -> Tuple[Dict, np.ndarray]:
    from sklearn.cluster import KMeans

    if features != "basic":
        raise ValueError("Only 'basic' features supported.")
    nodes, X = _role_features_basic(G)
    km = KMeans(n_clusters=int(n_roles), n_init=n_init, random_state=seed)
    lab = km.fit_predict(X)
    assign = {n: int(c) for n, c in zip(nodes, lab)}
    return assign, km.cluster_centers_
