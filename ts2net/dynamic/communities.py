"""
Temporal community tracking across graph windows.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def community_labels(G: nx.Graph) -> dict[int, int]:
    """
    Assign community ids using connected components (fallback for sparse graphs).

    For denser graphs with networkx >= 2.8, uses greedy modularity communities
    when available.
    """
    if G.number_of_nodes() == 0:
        return {}

    try:
        from networkx.algorithms.community import greedy_modularity_communities

        if G.number_of_edges() > 0:
            comms = list(greedy_modularity_communities(G))
            labels: dict[int, int] = {}
            for cid, comm in enumerate(comms):
                for node in comm:
                    labels[node] = cid
            return labels
    except Exception:
        pass

    labels = {}
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            labels[node] = cid
    return labels


def track_communities(
    graphs: list[nx.Graph],
) -> dict[str, object]:
    """
    Track community structure across graph windows.

    Returns
    -------
    dict
        ``labels_per_window`` (list of node->community dicts),
        ``n_communities`` (array), ``stability`` (mean Jaccard overlap
        of consecutive community partitions).
    """
    if not graphs:
        return {
            "labels_per_window": [],
            "n_communities": np.array([], dtype=np.int64),
            "stability": np.array([], dtype=np.float64),
        }

    labels_per_window = [community_labels(G) for G in graphs]
    n_communities = np.array(
        [len(set(lbls.values())) for lbls in labels_per_window],
        dtype=np.int64,
    )

    stability: list[float] = []
    for i in range(len(labels_per_window) - 1):
        a = labels_per_window[i]
        b = labels_per_window[i + 1]
        common = set(a) & set(b)
        if not common:
            stability.append(0.0)
            continue
        same = sum(1 for n in common if a[n] == b[n])
        stability.append(same / len(common))

    return {
        "labels_per_window": labels_per_window,
        "n_communities": n_communities,
        "stability": np.asarray(stability, dtype=np.float64),
    }
