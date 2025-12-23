"""
Unified graph visualization API for ts2net.

Provides TSGraph dataclass and builder functions for creating
visualization-ready graph objects with geometry and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
import networkx as nx


VisKind = Literal["hvg", "nvg", "bounded_nvg"]
WeightMode = Literal["none", "absdiff", "time_gap", "min_clearance", "slope"]


@dataclass(frozen=True)
class TSGraph:
    """Container for a time-series-derived graph plus geometry and build metadata.

    Attributes:
        graph: NetworkX graph with node and edge attributes.
        pos: Optional 2D coordinates for nodes. Keys match graph nodes.
            Defaults to (t, x[t]) for visibility graphs.
        meta: Build metadata such as method, parameters, and data shape.
    """
    graph: Union[nx.Graph, nx.DiGraph]
    pos: Optional[Dict[int, np.ndarray]]
    meta: Dict[str, Any]


def build_visibility_graph(
    x: Union[np.ndarray, Iterable[float]],
    *,
    kind: VisKind = "hvg",
    directed: bool = False,
    weighted: Union[bool, WeightMode] = False,
    weight_mode: Optional[WeightMode] = None,
    limit: Optional[int] = None,
    max_edges: Optional[int] = None,
    max_edges_per_node: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
    include_self_loops: bool = False,
    return_pos: bool = True,
    dtype: np.dtype = np.float64,
) -> TSGraph:
    """Construct HVG or NVG style graphs with optional direction and weights.

    Nodes map to time index i.
    Edge direction uses time forward orientation i -> j when i < j.
    Weights attach as edge attribute "weight".
    Distances and aux values attach as edge attributes when needed.

    Args:
        x: 1D series.
        kind: hvg, nvg, or bounded_nvg.
        directed: If True, emit a DiGraph and only time forward edges.
        weighted: False, True, or a string mode.
        weight_mode: Optional explicit mode. Overrides weighted when set.
        limit: Window limit for NVG variants.
        max_edges: Global cap for bounded_nvg.
        max_edges_per_node: Per node cap for bounded_nvg.
        max_memory_mb: Memory guard for bounded_nvg.
        include_self_loops: Rare. Default False.
        return_pos: If True, pos uses (t, x[t]) so plots match the series.
        dtype: Numeric dtype.

    Returns:
        TSGraph container with graph, pos, and meta.
    """
    x = np.asarray(x, dtype=dtype)
    n = int(x.shape[0])
    if n < 2:
        g = nx.DiGraph() if directed else nx.Graph()
        for i in range(n):
            g.add_node(i, t=i, x=float(x[i]))
        pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
        return TSGraph(graph=g, pos=pos, meta={"method": "visibility", "kind": kind})

    mode = _resolve_weight_mode(weighted=weighted, weight_mode=weight_mode)

    edges = _visibility_edges(
        x,
        kind=kind,
        limit=limit,
        max_edges=max_edges,
        max_edges_per_node=max_edges_per_node,
        max_memory_mb=max_memory_mb,
    )

    g: Union[nx.Graph, nx.DiGraph] = nx.DiGraph() if directed else nx.Graph()

    for i in range(n):
        g.add_node(i, t=i, x=float(x[i]))

    for i, j in edges:
        if not include_self_loops and i == j:
            continue
        u, v = (i, j) if not directed else ((i, j) if i < j else (j, i))
        if u == v and not include_self_loops:
            continue

        attrs: Dict[str, Any] = {}
        if mode != "none":
            attrs["weight"] = _vis_weight(x, i=u, j=v, mode=mode)
        g.add_edge(u, v, **attrs)

    pos = {i: np.array([i, x[i]], dtype=dtype) for i in range(n)} if return_pos else None
    meta = {
        "method": "visibility",
        "kind": kind,
        "directed": bool(directed),
        "weight_mode": mode,
        "limit": limit,
        "max_edges": max_edges,
        "max_edges_per_node": max_edges_per_node,
        "max_memory_mb": max_memory_mb,
        "n": n,
    }
    return TSGraph(graph=g, pos=pos, meta=meta)


def _resolve_weight_mode(*, weighted: Union[bool, WeightMode], weight_mode: Optional[WeightMode]) -> WeightMode:
    """Resolve weight mode from weighted flag and explicit weight_mode."""
    if weight_mode is not None:
        return weight_mode
    if weighted is True:
        return "absdiff"
    if weighted is False:
        return "none"
    return weighted


def _visibility_edges(
    x: np.ndarray,
    *,
    kind: VisKind,
    limit: Optional[int],
    max_edges: Optional[int],
    max_edges_per_node: Optional[int],
    max_memory_mb: Optional[int],
) -> np.ndarray:
    """Return array of undirected candidate edges as pairs (i, j)."""
    from ts2net.core.visibility.hvg import _hvg_edges_numba
    from ts2net.core.visibility.nvg import _nvg_edges_numba
    
    if kind == "hvg":
        rows, cols, _ = _hvg_edges_numba(x, weighted=False, limit=limit if limit else -1)
        edges = np.column_stack([rows, cols])
        # Normalize to (i, j) where i < j for consistency
        edges = np.sort(edges, axis=1)
        # Remove duplicates
        edges = np.unique(edges, axis=0)
        return edges
    
    if kind == "nvg":
        rows, cols, _, _ = _nvg_edges_numba(
            x, 
            weighted=False, 
            limit=limit if limit else -1,
            max_edges=max_edges if max_edges else -1,
            max_edges_per_node=max_edges_per_node if max_edges_per_node else -1
        )
        edges = np.column_stack([rows, cols])
        # Normalize to (i, j) where i < j for consistency
        edges = np.sort(edges, axis=1)
        # Remove duplicates
        edges = np.unique(edges, axis=0)
        return edges
    
    if kind == "bounded_nvg":
        # For now, use same as nvg with limits
        # TODO: Implement proper bounded_nvg with memory checks
        return _visibility_edges(x, kind="nvg", limit=limit, 
                                max_edges=max_edges, 
                                max_edges_per_node=max_edges_per_node,
                                max_memory_mb=max_memory_mb)
    
    raise ValueError(f"Unknown kind {kind}")


def _vis_weight(x: np.ndarray, *, i: int, j: int, mode: WeightMode) -> float:
    """Compute edge weight for visibility graph edge (i, j)."""
    if i == j:
        return 0.0
    if mode == "absdiff":
        return float(abs(x[j] - x[i]))
    if mode == "time_gap":
        return float(abs(j - i))
    if mode == "slope":
        return float((x[j] - x[i]) / (j - i))
    if mode == "min_clearance":
        return float(_min_clearance(x, i=i, j=j))
    if mode == "none":
        return 1.0
    raise ValueError(f"Unknown weight mode {mode}")


def _min_clearance(x: np.ndarray, *, i: int, j: int) -> float:
    """Compute minimum clearance between points i and j for visibility."""
    lo, hi = (i, j) if i < j else (j, i)
    if hi - lo <= 1:
        return float("inf")
    baseline = min(x[lo], x[hi])
    mid = x[lo + 1 : hi]
    if len(mid) == 0:
        return float("inf")
    return float(baseline - np.max(mid))
