"""
Backend-routed visibility graph builders and degree-only stats paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from .backend import resolve_compute_backend

_RUST_DEGREES: dict[str, Any] | None = None


def _rust_degree_fns() -> dict[str, Any]:
    global _RUST_DEGREES
    if _RUST_DEGREES is None:
        try:
            from ts2net_rs import hvg_degrees, nvg_degrees

            _RUST_DEGREES = {"hvg": hvg_degrees, "nvg": nvg_degrees}
        except ImportError:
            _RUST_DEGREES = {}
    return _RUST_DEGREES


def stats_from_degree_sequences(
    *,
    n_nodes: int,
    n_edges: int,
    degrees: NDArray[np.int64],
    directed: bool = False,
    in_degrees: NDArray[np.int64] | None = None,
    out_degrees: NDArray[np.int64] | None = None,
) -> dict[str, float]:
    """Build a Graph.summary-compatible stats dict from degree arrays."""
    if directed:
        assert in_degrees is not None and out_degrees is not None
        primary = out_degrees
    else:
        primary = degrees

    max_edges = n_nodes * (n_nodes - 1)
    if not directed:
        max_edges //= 2

    stats: dict[str, float] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": float(np.mean(primary)) if n_nodes else 0.0,
        "std_degree": float(np.std(primary)) if n_nodes > 1 else 0.0,
        "min_degree": int(np.min(primary)) if n_nodes else 0,
        "max_degree": int(np.max(primary)) if n_nodes else 0,
        "density": n_edges / max_edges if max_edges > 0 else 0.0,
    }

    if directed and in_degrees is not None and out_degrees is not None:
        total = in_degrees + out_degrees
        stats["avg_in_degree"] = float(np.mean(in_degrees))
        stats["std_in_degree"] = float(np.std(in_degrees)) if n_nodes > 1 else 0.0
        stats["avg_out_degree"] = float(np.mean(out_degrees))
        stats["std_out_degree"] = float(np.std(out_degrees)) if n_nodes > 1 else 0.0
        stats["min_in_degree"] = int(np.min(in_degrees))
        stats["max_in_degree"] = int(np.max(in_degrees))
        stats["min_out_degree"] = int(np.min(out_degrees))
        stats["max_out_degree"] = int(np.max(out_degrees))
        irr = np.zeros(n_nodes, dtype=np.float64)
        mask = total > 0
        irr[mask] = np.abs(in_degrees[mask] - out_degrees[mask]) / total[mask]
        stats["irreversibility_score"] = float(np.mean(irr))

    return stats


def _hvg_stats_numba(
    x: NDArray[np.float64],
    directed: bool,
    limit: int | None,
) -> dict[str, float]:
    from .visibility.hvg import _hvg_edges_numba

    limit_val = -1 if limit is None else int(limit)
    rows, cols, _ = _hvg_edges_numba(x, False, limit_val)
    n_nodes = len(x)
    if directed:
        in_d = np.zeros(n_nodes, dtype=np.int64)
        out_d = np.zeros(n_nodes, dtype=np.int64)
        for i, j in zip(rows, cols, strict=True):
            out_d[i] += 1
            in_d[j] += 1
        return stats_from_degree_sequences(
            n_nodes=n_nodes,
            n_edges=len(rows),
            degrees=out_d,
            directed=True,
            in_degrees=in_d,
            out_degrees=out_d,
        )
    deg = np.zeros(n_nodes, dtype=np.int64)
    for i, j in zip(rows, cols, strict=True):
        deg[i] += 1
        deg[j] += 1
    return stats_from_degree_sequences(
        n_nodes=n_nodes,
        n_edges=len(rows),
        degrees=deg,
    )


def _nvg_stats_numba(
    x: NDArray[np.float64],
    limit: int | None,
) -> dict[str, float]:
    from .visibility.nvg import _nvg_edges_numba

    limit_val = -1 if limit is None else int(limit)
    rows, cols, _, _ = _nvg_edges_numba(x, False, limit_val, -1, -1)
    n_nodes = len(x)
    deg = np.zeros(n_nodes, dtype=np.int64)
    for i, j in zip(rows, cols, strict=True):
        deg[i] += 1
        deg[j] += 1
    return stats_from_degree_sequences(
        n_nodes=n_nodes,
        n_edges=len(rows),
        degrees=deg,
    )


def visibility_degree_stats(
    x: NDArray[np.float64],
    method: str,
    *,
    directed: bool = False,
    limit: int | None = None,
    weighted: bool = False,
    backend: str = "auto",
) -> dict[str, float] | None:
    """
    Degree-only graph stats for HVG/NVG (no edge materialisation).

    Returns ``None`` when the fast path does not apply (e.g. weighted graphs).
    """
    if weighted:
        return None

    x = validate_series(x, f"{method}_degree_stats")
    method = method.lower()
    if method not in ("hvg", "nvg"):
        return None

    backend = resolve_compute_backend(backend)

    if backend == "rust":
        fns = _rust_degree_fns()
        if fns:
            if method == "hvg":
                raw = fns["hvg"](x, directed=directed, limit=limit)
                n_edges = int(raw["n_edges"])
                if directed:
                    in_d = np.asarray(raw["in_degree"], dtype=np.int64)
                    out_d = np.asarray(raw["out_degree"], dtype=np.int64)
                    return stats_from_degree_sequences(
                        n_nodes=len(x),
                        n_edges=n_edges,
                        degrees=out_d,
                        directed=True,
                        in_degrees=in_d,
                        out_degrees=out_d,
                    )
                deg = np.asarray(raw["degree"], dtype=np.int64)
                return stats_from_degree_sequences(
                    n_nodes=len(x), n_edges=n_edges, degrees=deg
                )
            raw = fns["nvg"](x, limit=limit)
            deg = np.asarray(raw["degree"], dtype=np.int64)
            return stats_from_degree_sequences(
                n_nodes=len(x), n_edges=int(raw["n_edges"]), degrees=deg
            )

    if backend in ("rust", "numba"):
        if method == "hvg":
            return _hvg_stats_numba(x, directed, limit)
        return _nvg_stats_numba(x, limit)

    if method == "hvg":
        return _hvg_stats_numba(x, directed, limit)
    return _nvg_stats_numba(x, limit)
