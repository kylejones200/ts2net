"""
Causal network metrics for directed graphs.

Provides path-based causality strength, per-node directionality indices,
and global summaries for networks built from transfer entropy or Granger tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np


def _edge_weight(G: nx.DiGraph, u: int, v: int, weight: str) -> float:
    data = G.get_edge_data(u, v, default={})
    return float(data.get(weight, 1.0))


def causal_strength(
    G: nx.DiGraph,
    source: int,
    target: int,
    weight: str = "weight",
) -> float:
    """
    Path-based causal strength from ``source`` to ``target``.

    Uses the maximum-weight directed path (log-sum of edge weights when
    weights are probabilities or TE values). Returns 0 when no path exists.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed causal network.
    source, target : int
        Node indices.
    weight : str, default "weight"
        Edge attribute used as path weight.

    Returns
    -------
    float
        Maximum path strength (product of edge weights along best path).
    """
    if source == target:
        return 0.0

    if not nx.has_path(G, source, target):
        return 0.0

    # Negate weights so max-product path becomes min-sum on -log(w)
    def edge_cost(u: int, v: int, data: dict) -> float:
        w = float(data.get(weight, 1.0))
        if w <= 0:
            return float("inf")
        return -np.log(w)

    try:
        path = nx.dijkstra_path(G, source, target, weight=edge_cost)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.0

    strength = 1.0
    for u, v in zip(path[:-1], path[1:], strict=True):
        strength *= _edge_weight(G, u, v, weight)
    return float(strength)


def directionality_index(
    G: nx.DiGraph,
    node: Optional[int] = None,
    weight: str = "weight",
) -> Union[float, Dict[int, float]]:
    """
    Net causal directionality for one or all nodes.

    DI(v) = (out_strength - in_strength) / (out_strength + in_strength)

    Values near +1 indicate net causal emitter; near -1 net receiver.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed causal network.
    node : int, optional
        Node index. If None, returns dict for all nodes.
    weight : str, default "weight"
        Edge attribute for strength.

    Returns
    -------
    float or dict[int, float]
        Directionality index in [-1, 1].
    """
    def _node_di(v: int) -> float:
        out_s = sum(_edge_weight(G, v, n, weight) for n in G.successors(v))
        in_s = sum(_edge_weight(G, n, v, weight) for n in G.predecessors(v))
        total = out_s + in_s
        if total == 0:
            return 0.0
        return float((out_s - in_s) / total)

    if node is not None:
        return _node_di(node)

    return {v: _node_di(v) for v in G.nodes()}


def causal_network_metrics(
    G: nx.DiGraph,
    weight: str = "weight",
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Summarize a directed causal network.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed causal network.
    weight : str, default "weight"
        Edge attribute for strength-based metrics.

    Returns
    -------
    dict
        Global metrics (density, avg_in_strength, avg_out_strength,
        avg_path_length on the largest weakly connected component) and
        per-node ``directionality`` indices.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "avg_in_strength": 0.0,
            "avg_out_strength": 0.0,
            "avg_path_length": np.nan,
            "directionality": {},
        }

    in_strength = {
        v: sum(_edge_weight(G, u, v, weight) for u in G.predecessors(v))
        for v in G.nodes()
    }
    out_strength = {
        v: sum(_edge_weight(G, v, w, weight) for w in G.successors(v))
        for v in G.nodes()
    }

    density = m / (n * (n - 1)) if n > 1 else 0.0

    avg_path_length = np.nan
    if m > 0:
        try:
            wcc = max(nx.weakly_connected_components(G), key=len)
            sub = G.subgraph(wcc).copy()
            if sub.number_of_edges() > 0 and sub.number_of_nodes() > 1:
                avg_path_length = float(
                    nx.average_shortest_path_length(sub, weight=weight)
                )
        except (nx.NetworkXError, ZeroDivisionError):
            avg_path_length = np.nan

    return {
        "n_nodes": n,
        "n_edges": m,
        "density": float(density),
        "avg_in_strength": float(np.mean(list(in_strength.values()))),
        "avg_out_strength": float(np.mean(list(out_strength.values()))),
        "avg_path_length": avg_path_length,
        "directionality": directionality_index(G, weight=weight),
        "top_emitters": _top_nodes(out_strength, k=min(5, n)),
        "top_receivers": _top_nodes(in_strength, k=min(5, n)),
    }


def _top_nodes(strength: Dict[int, float], k: int) -> List[int]:
    ranked = sorted(strength.items(), key=lambda item: item[1], reverse=True)
    return [node for node, val in ranked[:k] if val > 0]
