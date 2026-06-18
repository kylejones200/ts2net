"""
Stats-only transition network path without NetworkX materialisation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from .backend import resolve_compute_backend
from .visibility_backend import stats_from_degree_sequences


def _ordinal_pattern_indices(x: NDArray[np.float64], order: int) -> NDArray[np.int64]:
    n = len(x)
    patterns: list[tuple[int, ...]] = []
    for i in range(n - order + 1):
        subseq = x[i : i + order]
        patterns.append(tuple(np.argsort(subseq)))

    unique_patterns = list(dict.fromkeys(patterns))
    pattern_to_idx = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
    return np.array([pattern_to_idx[pattern] for pattern in patterns], dtype=np.int64)


def _digitize_ordinal(x: NDArray[np.float64], order: int) -> NDArray[np.int64]:
    return _ordinal_pattern_indices(x, order)


def _transition_counts(
    digitized: NDArray[np.int64],
    order: int,
) -> tuple[int, int, NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return node/edge counts and degree arrays (matches core transition transform)."""
    sequences = [
        digitized[i : i + order + 1]
        for i in range(len(digitized) - order)
    ]

    node_id: dict[tuple[int, ...], int] = {}
    in_deg: dict[int, int] = defaultdict(int)
    out_deg: dict[int, int] = defaultdict(int)
    edge_seen: set[tuple[int, int]] = set()

    def _nid(state: tuple[int, ...]) -> int:
        if state not in node_id:
            node_id[state] = len(node_id)
        return node_id[state]

    for seq in sequences:
        source = tuple(int(v) for v in seq[:-1])
        target = tuple(int(v) for v in seq[1:])
        si = _nid(source)
        ti = _nid(target)
        if (si, ti) not in edge_seen:
            edge_seen.add((si, ti))
            out_deg[si] += 1
            in_deg[ti] += 1
    n_edges = len(edge_seen)

    n_nodes = len(node_id)
    if n_nodes == 0:
        empty = np.zeros(0, dtype=np.int64)
        return 0, 0, empty, empty, empty

    in_d = np.array([in_deg.get(i, 0) for i in range(n_nodes)], dtype=np.int64)
    out_d = np.array([out_deg.get(i, 0) for i in range(n_nodes)], dtype=np.int64)
    total = in_d + out_d
    return n_nodes, n_edges, total, in_d, out_d


def transition_degree_stats(
    x: NDArray[np.float64],
    *,
    symbolizer: str = "ordinal",
    order: int = 3,
    backend: str = "auto",
) -> dict[str, float] | None:
    """
    Graph stats for transition networks without building a NetworkX graph.

    Currently supports ``symbolizer='ordinal'`` (the streaming default).
    """
    _ = resolve_compute_backend(backend)
    x = validate_series(x, "transition_degree_stats")
    if len(x) < order + 1:
        return None

    if symbolizer.lower() != "ordinal":
        return None

    digitized = _digitize_ordinal(x, order)
    n_nodes, n_edges, total, in_d, out_d = _transition_counts(digitized, order)
    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "avg_degree": 0.0,
            "std_degree": 0.0,
        }

    return stats_from_degree_sequences(
        n_nodes=n_nodes,
        n_edges=n_edges,
        degrees=total,
        directed=True,
        in_degrees=in_d,
        out_degrees=out_d,
    )
