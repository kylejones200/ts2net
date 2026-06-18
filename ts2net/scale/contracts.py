"""
Performance scaling contracts for ts2net graph builders.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceContract:
    """Expected time and memory scaling for a builder."""

    method: str
    time_complexity: str
    memory_complexity: str
    recommended_output: str
    notes: str = ""

    def summary(self) -> str:
        """One-line scaling summary."""
        return (
            f"{self.method}: time {self.time_complexity}, "
            f"memory {self.memory_complexity}; "
            f"recommended output={self.recommended_output}"
        )


_CONTRACTS: dict[str, PerformanceContract] = {
    "hvg": PerformanceContract(
        method="hvg",
        time_complexity="O(n)",
        memory_complexity="O(n) edges",
        recommended_output="stats",
        notes="Stack algorithm; safe for long series at full resolution.",
    ),
    "nvg": PerformanceContract(
        method="nvg",
        time_complexity="O(n·L) with horizon limit L",
        memory_complexity="O(n·L) edges",
        recommended_output="stats or degrees",
        notes="Use limit for n > 10k; default limit applied for large n.",
    ),
    "recurrence": PerformanceContract(
        method="recurrence",
        time_complexity="O(n²) exact; O(n·k) for kNN",
        memory_complexity="O(n²) dense or O(edges) sparse",
        recommended_output="stats",
        notes="Prefer rule='knn' for embedding length n > 5k.",
    ),
    "transition": PerformanceContract(
        method="transition",
        time_complexity="O(n)",
        memory_complexity="O(n + s²) for s symbols",
        recommended_output="stats",
        notes="Symbol count s is typically small (ordinal patterns).",
    ),
    "build_windows": PerformanceContract(
        method="build_windows",
        time_complexity="O(w·cost(method, window)) for w windows",
        memory_complexity="O(w) stats or O(w·window) if materialized",
        recommended_output="stats",
        notes="Use build_windows_streaming() to avoid materializing all windows.",
    ),
    "ts_dist": PerformanceContract(
        method="ts_dist",
        time_complexity="O(p²·n) for p series of length n",
        memory_complexity="O(p²)",
        recommended_output="distance matrix",
        notes="Use n_jobs=-1; ts_dist_part() for out-of-core panels.",
    ),
    "cdist_dtw": PerformanceContract(
        method="cdist_dtw",
        time_complexity="O(p²·n²) exact; O(p²·n·b) with Sakoe-Chiba band b",
        memory_complexity="O(p²) or O(chunk²) with cdist_dtw_chunked()",
        recommended_output="distance matrix",
        notes="Rust backend preferred; cdist_dtw_chunked() for panels ≥ 64 series.",
    ),
}


def get_performance_contract(method: str) -> PerformanceContract:
    """
    Return documented scaling behavior for a builder or API entry point.

    Parameters
    ----------
    method : str
        Builder or API name (e.g. ``hvg``, ``build_windows``, ``ts_dist``).

    Returns
    -------
    PerformanceContract
        Time/memory complexity and usage notes.

    Raises
    ------
    KeyError
        If no contract is registered for ``method``.
    """
    key = method.lower()
    if key not in _CONTRACTS:
        known = ", ".join(sorted(_CONTRACTS))
        raise KeyError(f"Unknown method {method!r}. Known contracts: {known}")
    return _CONTRACTS[key]


def list_performance_contracts() -> dict[str, PerformanceContract]:
    """Return all registered performance contracts."""
    return dict(_CONTRACTS)
