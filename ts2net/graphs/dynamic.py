"""
Dynamic graph sequences from rolling windows.

Builds graph sequences over time and tracks edge birth, death, and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int, validate_series
from ..api import build_network
from ..multivariate.windows import ts_to_windows

BuilderMethod = Literal["hvg", "nvg", "recurrence", "transition"]


def _edge_set(G: nx.Graph, undirected: bool = True) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for u, v in G.edges():
        if undirected:
            edges.add((min(u, v), max(u, v)))
        else:
            edges.add((u, v))
    return edges


def edge_birth_death(
    G_prev: nx.Graph,
    G_next: nx.Graph,
    undirected: bool = True,
) -> dict[str, set[tuple[int, int]] | int]:
    """
    Compare consecutive graphs and identify edge births and deaths.

    Returns
    -------
    dict
        Keys: ``born``, ``died``, ``persisted``, ``n_births``, ``n_deaths``.
    """
    prev = _edge_set(G_prev, undirected=undirected)
    nxt = _edge_set(G_next, undirected=undirected)
    born = nxt - prev
    died = prev - nxt
    persisted = prev & nxt
    return {
        "born": born,
        "died": died,
        "persisted": persisted,
        "n_births": len(born),
        "n_deaths": len(died),
        "n_persisted": len(persisted),
    }


def graph_churn(
    graphs: list[nx.Graph],
    undirected: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """
    Edge churn metrics across a graph sequence.

    Returns arrays of length ``len(graphs) - 1`` for births, deaths, and
    Jaccard similarity between consecutive snapshots.
    """
    if len(graphs) < 2:
        return {
            "births": np.array([], dtype=np.float64),
            "deaths": np.array([], dtype=np.float64),
            "jaccard": np.array([], dtype=np.float64),
        }

    births, deaths, jaccard = [], [], []
    for i in range(len(graphs) - 1):
        diff = edge_birth_death(graphs[i], graphs[i + 1], undirected=undirected)
        births.append(diff["n_births"])
        deaths.append(diff["n_deaths"])
        prev = _edge_set(graphs[i], undirected=undirected)
        nxt = _edge_set(graphs[i + 1], undirected=undirected)
        union = prev | nxt
        jaccard.append(len(prev & nxt) / len(union) if union else 1.0)

    return {
        "births": np.asarray(births, dtype=np.float64),
        "deaths": np.asarray(deaths, dtype=np.float64),
        "jaccard": np.asarray(jaccard, dtype=np.float64),
    }


def edge_persistence(
    graphs: list[nx.Graph],
    undirected: bool = True,
) -> dict[tuple[int, int], float]:
    """
    Fraction of windows each edge appears in.

    Returns
    -------
    dict
        Edge -> persistence score in [0, 1].
    """
    if not graphs:
        return {}

    n = len(graphs)
    counts: dict[tuple[int, int], int] = {}
    for G in graphs:
        for edge in _edge_set(G, undirected=undirected):
            counts[edge] = counts.get(edge, 0) + 1
    return {edge: count / n for edge, count in counts.items()}


@dataclass
class RollingGraphSequence:
    """
      Graph sequence built from sliding windows over one time series.

      Parameters
      ----------
      method : str
          Builder: ``hvg``, ``nvg``, ``recurrence``, ``transition``.
      window, step : int
          Sliding window width and step.
      output : str
          Builder output mode (``stats``, ``degrees``, ``edges``).
    builder_kwargs
          Extra arguments for the network builder.
    """

    method: BuilderMethod = "hvg"
    window: int = 50
    step: int = 1
    output: str = "stats"
    builder_kwargs: dict[str, Any] = field(default_factory=dict)
    graphs_nx: list[nx.Graph] = field(default_factory=list)
    stats: list[dict[str, Any]] = field(default_factory=list)
    window_starts: NDArray[np.int64] = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_series(
        cls,
        x: NDArray[np.float64],
        window: int,
        step: int = 1,
        method: BuilderMethod = "hvg",
        output: str = "stats",
        as_networkx: bool = True,
        **builder_kwargs,
    ) -> RollingGraphSequence:
        """Build a rolling graph sequence from a univariate series."""
        x = validate_series(x, "RollingGraphSequence")
        window = validate_positive_int("window", window)
        step = validate_positive_int("step", step)

        windows = ts_to_windows(x, width=window, by=step)
        seq = cls(
            method=method,
            window=window,
            step=step,
            output=output,
            builder_kwargs=builder_kwargs,
        )

        starts = list(range(0, len(x) - window + 1, step))
        seq.window_starts = np.asarray(starts, dtype=np.int64)

        for w in windows:
            builder = build_network(w, method, output=output, **builder_kwargs)
            seq.stats.append(builder.stats())
            if as_networkx:
                seq.graphs_nx.append(builder.as_networkx(force=True))

        return seq

    def churn(self) -> dict[str, NDArray[np.float64]]:
        """Edge churn between consecutive graphs."""
        return graph_churn(self.graphs_nx)

    def persistence(self) -> dict[tuple[int, int], float]:
        """Edge persistence across all windows."""
        return edge_persistence(self.graphs_nx)

    def stat_series(self, key: str) -> NDArray[np.float64]:
        """Extract one summary statistic across windows."""
        return np.asarray([s.get(key, np.nan) for s in self.stats], dtype=np.float64)
