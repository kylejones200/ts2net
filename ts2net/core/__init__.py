"""
Core module for time series to network conversion.

This module provides implementations of various time series to network conversion
algorithms including Recurrence Networks, Visibility Graphs, and Transition Networks.
"""

from __future__ import annotations

# Standard library imports
import math
import random
from dataclasses import dataclass, fields
from itertools import combinations
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union, Callable

# Third-party imports
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix

# Local imports
from .recurrence import RecurrenceNetwork
from .transition import TransitionNetwork
from .visibility import HVG, NVG

# Removed circular import - these classes are defined in this file
try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        """Dummy decorator when numba is not available."""

        def deco(f):
            return f

        return deco


# Third-party imports
try:
    from scipy.sparse import csr_matrix
except ImportError:
    csr_matrix = None

try:
    from networkx import from_scipy_sparse_array as _nx_from_csr
except ImportError:
    _nx_from_csr = None

try:
    from networkx import from_scipy_sparse_matrix as _nx_from_csr_legacy
except ImportError:
    _nx_from_csr_legacy = None

# Local application imports - temporarily disable Rust imports due to build issues
try:
    from .core_rust import (
        hvg_graph as _hvg_graph_rs,
        knn as _knn_rs,
        nvg_graph as _nvg_graph_rs,
        radius as _radius_rs,
        rn_adj_epsilon as _rn_adj_eps_rs,
    )
    _rust_available = True
except ImportError:
    # Rust extension not available, use Python fallbacks
    _hvg_graph_rs = None
    _knn_rs = None
    _nvg_graph_rs = None
    _radius_rs = None
    _rn_adj_eps_rs = None
    _rust_available = False
from .stats_summary import GraphSummaryResult

# SKMixin is a simple mixin class for scikit-learn compatibility
class SKMixin:
    """Mixin class for scikit-learn compatibility."""
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {k: v for k, v in self.__dict__.items() if not k.endswith('_')}
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# HVG and NVG implementations have been moved to their respective modules
# in the visibility/ subpackage


# ---- Weighted visibility helpers ----
def _vis_weights(y: np.ndarray, E: np.ndarray, mode: str) -> np.ndarray:
    _WEIGHT_MODES = {
        "absdiff": lambda y, E: np.abs(y[E[:, 0]] - y[E[:, 1]]).astype(float),
        "inv_dist": lambda y, E: 1.0 / (np.abs(E[:, 1] - E[:, 0]).astype(float)),
        "slope": lambda y, E: (y[E[:, 1]] - y[E[:, 0]]) / (E[:, 1] - E[:, 0]),
        "distance": lambda y, E: np.abs(E[:, 1] - E[:, 0]).astype(float),
        "product": lambda y, E: (y[E[:, 0]] * y[E[:, 1]]).astype(float),
        "mean": lambda y, E: ((y[E[:, 0]] + y[E[:, 1]]) / 2.0).astype(float),
    }
    
    weight_fn = _WEIGHT_MODES.get(mode)
    if weight_fn is None:
        raise ValueError(f"Unknown mode: {mode}")
    
    return weight_fn(y, E)


def _graph_from_edges_weighted(
    n: int, E: np.ndarray, w: np.ndarray, sparse: bool
) -> Tuple[nx.Graph, object]:
    if sparse and csr_matrix is not None:
        rows = E[:, 0].astype(np.int64)
        cols = E[:, 1].astype(np.int64)
        A = csr_matrix((w.astype(float), (rows, cols)), shape=(n, n))
        A = A.maximum(A.T)
        A.setdiag(0.0)
        A.eliminate_zeros()
        G = nx.from_scipy_sparse_array(A)
        return G, A
    A = np.zeros((n, n), float)
    for (i, j), ww in zip(E, w):
        A[i, j] = ww
        A[j, i] = ww
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    return G, A


def _ordinal_patterns(x: np.ndarray, order: int) -> np.ndarray:
    n = x.size
    if order < 2:
        raise ValueError("order must be >= 2")
    L = n - order + 1
    if L <= 0:
        raise ValueError("Series too short for chosen order.")
    pats = np.empty(L, dtype=np.int64)
    base = np.arange(order)
    for i in range(L):
        w = x[i : i + order]
        idx = np.lexsort((base, w))
        code = 0
        used = np.zeros(order, dtype=np.int64)
        for j in range(order):
            r = idx[j]
            code += (r - used[:r].sum()) * math.factorial(order - j - 1)
            used[r] = 1
        pats[i] = code
    return pats


def _adj_to_graph(A: np.ndarray, directed: bool = False) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    n = A.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = np.nonzero(A)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        if directed:
            G.add_edge(i, j, weight=float(A[i, j]))
        else:
            if i < j:
                G.add_edge(i, j, weight=float(A[i, j]))
    return G


@njit(cache=True)
def _hvg_edges(y: np.ndarray) -> np.ndarray:
    n = y.size
    ei = []
    ej = []
    stack = []
    for j in range(n):
        while stack and y[stack[-1]] < y[j]:
            i = stack.pop()
            ei.append(i)
            ej.append(j)
        if stack:
            i = stack[-1]
            ei.append(i)
            ej.append(j)
        stack.append(j)
    m = len(ei)
    E = np.empty((m, 2), dtype=np.int64)
    for k in range(m):
        E[k, 0] = ei[k]
        E[k, 1] = ej[k]
    return E


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


def _giant_component(H: nx.Graph) -> nx.Graph:
    if H.number_of_nodes() == 0:
        return H
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    return H.subgraph(next(iter([c for c in comps]))).copy()


def small_world_summary(G: Union[nx.Graph, nx.DiGraph]) -> dict:
    H = G.to_undirected() if G.is_directed() else G
    GCC = _giant_component(H)
    n = GCC.number_of_nodes()
    m = GCC.number_of_edges()
    if n <= 1 or m == 0:
        return {
            "C": np.nan,
            "L": np.nan,
            "C_er": np.nan,
            "L_er": np.nan,
            "sigma": np.nan,
        }
    k_bar = 2.0 * m / n
    C = nx.average_clustering(GCC)
    try:
        L = nx.average_shortest_path_length(GCC)
    except nx.NetworkXError:
        L = np.nan
    p = (2.0 * m) / (n * (n - 1))
    C_er = p
    if k_bar > 1.0:
        L_er = math.log(n) / math.log(k_bar)
    else:
        L_er = np.nan
    num = (C / C_er) if C_er > 0 else np.nan
    den = (L / L_er) if (L_er is not None and not np.isnan(L_er)) else np.nan
    sigma = (
        num / den if (not np.isnan(num) and not np.isnan(den) and den != 0) else np.nan
    )
    return {"C": C, "L": L, "C_er": C_er, "L_er": L_er, "sigma": sigma}


def _sampled_combinations(nodes, r, max_samples=None, seed=3363):
    nodes = list(nodes)
    if max_samples is None:
        for combo in combinations(nodes, r):
            yield combo
        return
    rng = random.Random(seed)
    n = len(nodes)
    seen = set()
    attempts = 0
    cap = min(max_samples, 1_000_000)
    while len(seen) < cap and attempts < cap * 10:
        combo = tuple(sorted(rng.sample(nodes, r)))
        if combo not in seen:
            seen.add(combo)
            yield combo
        attempts += 1


def directed_3node_motifs(
    G: nx.DiGraph, max_samples: Optional[int] = None, seed: int = 3363
) -> dict:
    if not G.is_directed():
        raise ValueError("Graph must be directed.")
    counts = {}
    total = 0
    for trio in _sampled_combinations(G.nodes(), 3, max_samples, seed):
        sub = G.subgraph(trio).copy()
        if not nx.is_weakly_connected(sub):
            continue
        mapping = dict(zip(sub.nodes(), range(3)))
        sub = nx.relabel_nodes(sub, mapping)
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


def _deg_seq(sub: nx.Graph) -> List[int]:
    return sorted([d for _, d in sub.degree()])


def undirected_4node_motifs(
    G: nx.Graph, max_samples: Optional[int] = None, seed: int = 3363
) -> dict:
    if G.is_directed():
        raise ValueError("Graph must be undirected.")
    out = {"K4": 0, "C4": 0, "triangle_tail": 0, "P4": 0, "K1,3": 0, "other": 0}
    total = 0
    for quad in _sampled_combinations(G.nodes(), 4, max_samples, seed):
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


def motif_counts(
    G: Union[nx.Graph, nx.DiGraph],
    motif_type: str = '3node',
    max_samples: Optional[int] = None,
    seed: int = 3363,
) -> Dict[str, int]:
    """
    Count motifs in a graph.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    motif_type : str, default '3node'
        Type of motifs to count ('3node' or '4node')
    max_samples : int, optional
        Maximum number of samples for motif counting
    seed : int, default 3363
        Random seed for sampling
        
    Returns
    -------
    Dict[str, int]
        Dictionary of motif counts
    """
    if motif_type == '3node' and isinstance(G, nx.DiGraph):
        return directed_3node_motifs(G, max_samples, seed)
    elif motif_type == '4node' and isinstance(G, nx.Graph):
        return undirected_4node_motifs(G, max_samples, seed)
    else:
        return {}


def graph_summary(
    G: Union[nx.Graph, nx.DiGraph],
    motifs: Optional[str] = None,
    motif_samples: Optional[int] = None,
    seed: int = 3363,
) -> dict:
    """
    Compute a comprehensive summary of graph properties.
    
    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph
    motifs : str, optional
        Type of motifs to compute ('3node', '4node', or None)
    motif_samples : int, optional
        Maximum number of samples for motif counting
    seed : int, default 3363
        Random seed for sampling
        
    Returns
    -------
    dict
        Dictionary of graph properties
    """
    und = G.to_undirected() if G.is_directed() else G
    deg = dict(und.degree())
    deg_vals = np.array(list(deg.values()), dtype=float)
    
    out = {
        "n": und.number_of_nodes(),
        "m": und.number_of_edges(),
        "density": nx.density(und),
        "deg_mean": float(deg_vals.mean()) if deg_vals.size else 0.0,
        "deg_std": float(deg_vals.std(ddof=1)) if deg_vals.size > 1 else 0.0,
        "avg_degree": float(deg_vals.mean()) if deg_vals.size else 0.0,
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


def _as_1d(x: Union[np.ndarray, Iterable[float]]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("Expected a 1-D array.")
    return a


def _run_single(series: np.ndarray, builder: str, kwargs: dict):
    _BUILDERS = {
        "HVG": lambda s, kw: HVG().fit_transform(s),
        "NVG": lambda s, kw: NVG().fit_transform(s),
        "RN": lambda s, kw: RecurrenceNetwork(**kw).fit_transform(s),
        "TN": lambda s, kw: TransitionNetwork(**kw).fit_transform(s),
    }
    
    builder_fn = _BUILDERS.get(builder)
    if builder_fn is None:
        raise ValueError(f"Unknown builder: {builder}")
    
    return builder_fn(series, kwargs)


# Builder registry for batch processing
_BUILDERS = {
    'hvg': lambda x, **kwargs: HVG(**kwargs).fit_transform(x)[0],
    'nvg': lambda x, **kwargs: NVG(**kwargs).fit_transform(x)[0],
    'recurrence': lambda x, **kwargs: RecurrenceNetwork(**kwargs).fit_transform(x)[0],
    'transition': lambda x, **kwargs: TransitionNetwork(**kwargs).fit_transform(x)[0],
}


def batch_transform(
    X: Sequence[np.ndarray], builder: str, **kwargs
) -> List[Union[nx.Graph, nx.DiGraph]]:
    """
    Transform multiple time series using the specified builder.
    
    Parameters
    ----------
    X : Sequence[np.ndarray]
        Sequence of time series arrays
    builder : str
        Name of the builder to use
    **kwargs
        Additional keyword arguments for the builder
        
    Returns
    -------
    List[Union[nx.Graph, nx.DiGraph]]
        List of transformed graphs
    """
    return [_BUILDERS[builder](x, **kwargs) for x in X]
