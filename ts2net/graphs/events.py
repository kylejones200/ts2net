"""
Event-based temporal graph builders.

Detects events in time series and constructs networks from event timing
or cross-series event synchronization.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int, validate_series
from ..events import events_from_ts, tssim_event_sync
from ..multivariate.builders import net_knn, net_weighted
from .correlation import _correlation_to_distance

EdgeRule = Literal["consecutive", "window"]


def event_sequence_network(
    x: NDArray[np.float64],
    *,
    method: str = "peaks",
    thresh: float | None = None,
    min_separation: int = 1,
    edge_rule: EdgeRule = "window",
    max_interval: int = 10,
) -> tuple[nx.Graph, NDArray[np.int64]]:
    """
    Build a network whose nodes are detected events.

    Parameters
    ----------
    x : array (n,)
        Input time series.
    method : str
        Event detection: ``threshold`` or ``peaks`` (see ``events_from_ts``).
    thresh : float, optional
        Detection threshold.
    min_separation : int
        Minimum samples between events.
    edge_rule : {"consecutive", "window"}
        ``consecutive`` links adjacent events; ``window`` links all pairs within
        ``max_interval``.
    max_interval : int
        Maximum time gap for ``edge_rule="window"``.

    Returns
    -------
    G : networkx.Graph
        Nodes are event indices with ``time`` attribute.
    events : array
        Event time indices.
    """
    x = validate_series(x, "event_sequence_network")
    max_interval = validate_positive_int("max_interval", max_interval)

    events = events_from_ts(
        x,
        method=method,
        thresh=thresh,
        min_separation=min_separation,
    )

    G = nx.Graph()
    for idx, t in enumerate(events):
        G.add_node(int(idx), time=int(t))

    if len(events) < 2:
        return G, events

    if edge_rule == "consecutive":
        for i in range(len(events) - 1):
            gap = int(events[i + 1] - events[i])
            G.add_edge(i, i + 1, weight=float(gap))
    elif edge_rule == "window":
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                gap = int(events[j] - events[i])
                if gap <= max_interval:
                    G.add_edge(i, j, weight=float(gap))
    else:
        raise ValueError(f"Unknown edge_rule: {edge_rule}")

    return G, events


def event_sync_network(
    X: NDArray[np.float64],
    *,
    method: str = "peaks",
    thresh: float | None = None,
    min_separation: int = 1,
    adaptive: bool = True,
    rule: str = "knn",
    k: int = 3,
    threshold: float = 0.3,
) -> tuple[nx.Graph, NDArray[np.float64], list[NDArray[np.int64]]]:
    """
    Multivariate event synchronization network.

    Nodes represent time series; edge weight is event synchronization ``q``
  from ``tssim_event_sync``. Distance = ``1 - q``.

    Parameters
    ----------
    X : array (n_series, n_points)
        Panel of series.
    method, thresh, min_separation
        Event detection parameters per series.
    adaptive : bool
        Adaptive sync window in ``tssim_event_sync``.
    rule : {"knn", "threshold", "complete"}
        Network sparsification.
    k, threshold
        Sparsification parameters.

    Returns
    -------
    G : networkx.Graph
    sync_matrix : array (n_series, n_series) synchronization strengths
    event_sets : list of event index arrays per series
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_series = X.shape[0]
    event_sets: list[NDArray[np.int64]] = []
    for i in range(n_series):
        ev = events_from_ts(
            X[i],
            method=method,
            thresh=thresh,
            min_separation=min_separation,
        )
        event_sets.append(ev)

    sync = np.eye(n_series, dtype=np.float64)
    for i in range(n_series):
        for j in range(i + 1, n_series):
            _, _, q = tssim_event_sync(
                event_sets[i],
                event_sets[j],
                adaptive=adaptive,
            )
            sync[i, j] = sync[j, i] = float(q)

    D = _correlation_to_distance(sync)  # 1 - |sync|

    if rule == "knn":
        G, _ = net_knn(D, k=validate_positive_int("k", k), weighted=True)
    elif rule == "threshold":
        mask = sync >= threshold
        np.fill_diagonal(mask, False)
        D_thr = np.where(mask, D, 0.0)
        G, _ = net_weighted(D_thr, directed=False)
    elif rule == "complete":
        G, _ = net_weighted(D, directed=False)
    else:
        raise ValueError(f"Unknown rule: {rule}")

    for u, v in G.edges():
        G[u][v]["sync"] = float(sync[u, v])
        G[u][v]["weight"] = float(sync[u, v])

    return G, sync, event_sets
