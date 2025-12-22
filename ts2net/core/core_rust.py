# ts2net/core/core_rust.py - Rust bindings for performance-critical functions
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import networkx as nx

try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None

from ts2net_rs import (
    rn_adj_epsilon as _rn_adj_eps_rs,
    event_sync as _event_sync_rs,
    false_nearest_neighbors as _fnn_rs,
    cao_e1_e2 as _cao_rs,
    triangles_per_node as _tri_per_node_rs,
    clustering_avg as _clust_avg_rs,
    mean_shortest_path as _mspl_rs,
    surrogate_phase as _surr_phase_rs,
    iaaft as _iaaft_rs,
    corr_perm as _corr_perm_rs,
    moran_i as _moran_rs,
    hvg_edges as _hvg_edges_rs,
    nvg_edges_sweepline as _nvg_edges_rs,
    cdist_dtw as _cdist_dtw_rs,
    knn as _knn_rs,
    radius as _radius_rs,
)


def rn_adj_epsilon(
    X: np.ndarray, eps: float, metric: str = "euclidean", theiler: int = 0
) -> np.ndarray:
    X = np.asarray(X, float)
    A = _rn_adj_eps_rs(X, float(eps), metric, int(theiler))
    return np.array(A, dtype=np.uint8)


def event_sync(
    e1: np.ndarray, e2: np.ndarray, adaptive: bool = True, tau_max: Optional[float] = None
) -> Dict[str, Any]:
    e1 = np.asarray(e1, np.uint64)
    e2 = np.asarray(e2, np.uint64)
    out = np.array(
        _event_sync_rs(
            e1, e2, bool(adaptive), None if tau_max is None else float(tau_max)
        ),
        dtype=float,
    )
    c12, c21, ties, q12, q21, Q, n_delays = out[:7]
    delays = out[7:] if int(n_delays) > 0 else np.array([], float)
    return {
        "c12": c12,
        "c21": c21,
        "ties": ties,
        "q12": q12,
        "q21": q21,
        "Q": Q,
        "delays": delays,
    }


def fnn(
    x: np.ndarray, m_max: int = 10, tau: int = 1, rtol: float = 10.0, atol: float = 2.0
) -> np.ndarray:
    return np.array(
        _fnn_rs(np.asarray(x, float), int(m_max), int(tau), float(rtol), float(atol)),
        dtype=float,
    )


def cao(x: np.ndarray, m_max: int = 10, tau: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    E1, E2 = _cao_rs(np.asarray(x, float), int(m_max), int(tau))
    return np.array(E1, float), np.array(E2, float)


def triangles_per_node(n: int, edges: np.ndarray) -> np.ndarray:
    return np.array(
        _tri_per_node_rs(int(n), np.asarray(edges, np.uint64)), dtype=np.int64
    )


def clustering_avg(n: int, edges: np.ndarray) -> float:
    return float(_clust_avg_rs(int(n), np.asarray(edges, np.uint64)))


def mean_shortest_path(n: int, edges: np.ndarray) -> float:
    return float(_mspl_rs(int(n), np.asarray(edges, np.uint64)))


def surrogate_phase(x: np.ndarray, seed: int = 3363) -> np.ndarray:
    return np.array(_surr_phase_rs(np.asarray(x, float), int(seed)), dtype=float)


def iaaft(x: np.ndarray, iters: int = 50, seed: int = 3363) -> np.ndarray:
    return np.array(_iaaft_rs(np.asarray(x, float), int(iters), int(seed)), dtype=float)


def corr_perm(
    x: np.ndarray, y: np.ndarray, n_perm: int = 1000, seed: int = 3363
) -> float:
    return float(
        _corr_perm_rs(
            np.asarray(x, float), np.asarray(y, float), int(n_perm), int(seed)
        )
    )


def moran_i(y: np.ndarray, W: np.ndarray) -> Tuple[float, float]:
    I, z = _moran_rs(np.asarray(y, float), np.asarray(W, float))
    return float(I), float(z)


def _adj_from_edges(n: int, E: np.ndarray, directed: bool, sparse_out: bool):
    # Always use sparse to avoid memory blowup
    # Safety guardrail: refuse dense for large graphs
    if not sparse_out and n > 50_000:
        raise ValueError(
            f"Refusing to build dense adjacency matrix for n={n} nodes. "
            f"This would require ~{n**2 * 8 / 1e9:.1f} GB of memory. "
            f"Use sparse_out=True instead."
        )
    
    if csr_matrix is not None:
        rows = E[:, 0].astype(np.int64)
        cols = E[:, 1].astype(np.int64)
        data = np.ones(len(rows), dtype=float)
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        if not directed:
            A = A.maximum(A.T)
        A.setdiag(0.0)
        A.eliminate_zeros()
        return A
    
    # Fallback only for small graphs (should rarely be used)
    if n > 10_000:
        raise ValueError(
            f"scipy.sparse not available and n={n} is too large for dense matrix. "
            f"Install scipy to use sparse matrices."
        )
    A = np.zeros((n, n), dtype=int)
    for i, j in E:
        A[i, j] = 1
        if not directed:
            A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def _adj_to_graph(A, directed: bool = False):
    if hasattr(A, "tocsr"):
        return nx.from_scipy_sparse_array(
            A, create_using=nx.DiGraph() if directed else nx.Graph()
        )
    return (
        nx.from_numpy_array(A)
        if not directed
        else nx.from_numpy_array(A, create_using=nx.DiGraph())
    )


def hvg_graph(x: np.ndarray, sparse_out: bool = False):
    y = np.asarray(x, float).ravel()
    E = np.array(_hvg_edges_rs(y), dtype=np.int64)
    A = _adj_from_edges(len(y), E, directed=False, sparse_out=sparse_out)
    G = _adj_to_graph(A, directed=False)
    return G, A


def nvg_graph(x: np.ndarray, sparse_out: bool = False):
    y = np.asarray(x, float).ravel()
    E = np.array(_nvg_edges_rs(y), dtype=np.int64)
    A = _adj_from_edges(len(y), E, directed=False, sparse_out=sparse_out)
    G = _adj_to_graph(A, directed=False)
    return G, A


def cdist_dtw(X: np.ndarray, band: Optional[int] = None) -> np.ndarray:
    X = np.asarray(X, float)
    return np.array(_cdist_dtw_rs(X, band))


def knn(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, float)
    idx, dist = _knn_rs(X, int(k))
    return np.array(idx, dtype=np.int64), np.array(dist, dtype=float)


def radius(X: np.ndarray, eps: float) -> List[List[int]]:
    X = np.asarray(X, float)
    neighs = _radius_rs(X, float(eps))
    return [[int(j) for j in row] for row in neighs]
