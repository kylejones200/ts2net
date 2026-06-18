"""
Backend-routed recurrence network stats (degree-only, no edge materialisation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from .backend import resolve_compute_backend
from .visibility_backend import stats_from_degree_sequences


def _phase_points(
    x: NDArray[np.float64],
    m: int | None,
    tau: int,
) -> NDArray[np.float64]:
    """Phase-space points for recurrence (matches current core RecurrenceNetwork)."""
    x = validate_series(x, "recurrence_degree_stats")
    # Takens embedding is not applied in the core recurrence transform today.
    _ = m, tau
    return x.reshape(-1, 1)


def _stats_from_binary_adj(A: NDArray[np.uint8]) -> dict[str, float]:
    deg = A.sum(axis=1).astype(np.int64)
    n_edges = int(A.sum()) // 2
    return stats_from_degree_sequences(
        n_nodes=len(deg),
        n_edges=n_edges,
        degrees=deg,
    )


def recurrence_degree_stats(
    x: NDArray[np.float64],
    *,
    rule: str = "knn",
    k: int = 5,
    epsilon: float = 0.1,
    m: int | None = None,
    tau: int = 1,
    metric: str = "euclidean",
    backend: str = "auto",
) -> dict[str, float] | None:
    """
    Degree-only graph stats for recurrence networks (no edge materialisation).

    Returns ``None`` when the Rust fast path does not apply.
    """
    if metric != "euclidean":
        return None

    x = validate_series(x, "recurrence_degree_stats")
    if len(x) < 2:
        return None

    if resolve_compute_backend(backend) != "rust":
        return None

    try:
        from .core_rust import knn as rust_knn
        from .core_rust import rn_adj_epsilon
    except ImportError:
        return None

    pts = _phase_points(x, m, tau)
    n, d = pts.shape
    if d > 6:
        return None

    rule_key = (rule or "knn").lower()

    if rule_key == "knn":
        k_val = int(k)
        if k_val < 1 or k_val >= n:
            return None
        idx, _ = rust_knn(pts, k_val)
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for t in range(k_val):
                j = int(idx[i, t])
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return _stats_from_binary_adj(adj)

    if rule_key in ("epsilon", "radius"):
        if epsilon <= 0:
            return None
        adj = rn_adj_epsilon(pts, float(epsilon), metric="euclidean")
        return _stats_from_binary_adj(adj)

    return None
