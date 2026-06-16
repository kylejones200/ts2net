"""
Node role evolution across graph windows.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np

NodeRole = Literal["hub", "bridge", "peripheral", "isolate"]


def node_roles(
    G: nx.Graph,
    hub_quantile: float = 0.9,
    isolate_max_degree: int = 0,
) -> dict[int, NodeRole]:
    """
    Classify nodes by degree and betweenness centrality.

    Parameters
    ----------
    G : networkx.Graph
        Graph snapshot.
    hub_quantile : float, default 0.9
        Degree quantile above which a node is a hub.
    isolate_max_degree : int, default 0
        Maximum degree for isolate classification.

    Returns
    -------
    dict
        Node id -> role label.
    """
    if G.number_of_nodes() == 0:
        return {}

    degrees = dict(G.degree())
    deg_vals = np.array(list(degrees.values()), dtype=np.float64)
    hub_threshold = float(np.quantile(deg_vals, hub_quantile)) if len(deg_vals) else 0.0

    try:
        btw = nx.betweenness_centrality(G)
    except Exception:
        btw = {n: 0.0 for n in G.nodes()}

    btw_vals = np.array(list(btw.values()), dtype=np.float64)
    bridge_threshold = float(np.quantile(btw_vals, 0.75)) if len(btw_vals) else 0.0

    roles: dict[int, NodeRole] = {}
    for node in G.nodes():
        d = degrees.get(node, 0)
        if d <= isolate_max_degree:
            roles[node] = "isolate"
        elif d >= hub_threshold and hub_threshold > 0:
            roles[node] = "hub"
        elif btw.get(node, 0.0) >= bridge_threshold and bridge_threshold > 0:
            roles[node] = "bridge"
        else:
            roles[node] = "peripheral"
    return roles


def node_role_evolution(
    graphs: list[nx.Graph],
    **role_kwargs,
) -> dict[int, list[NodeRole | None]]:
    """
    Track node role labels across a graph sequence.

    Returns
    -------
    dict
        Node id -> list of roles (one per window; ``None`` if absent).
    """
    trajectories: dict[int, list[NodeRole | None]] = {}
    for G in graphs:
        roles = node_roles(G, **role_kwargs)
        all_nodes = set(trajectories) | set(G.nodes())
        for node in all_nodes:
            if node not in trajectories:
                trajectories[node] = []
            trajectories[node].append(roles.get(node))
    return trajectories
