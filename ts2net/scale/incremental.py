"""
Incremental graph updates for streaming time series (horizon 0.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.graph import Graph


def _hvg_visible(values: list[float], i: int, j: int) -> bool:
    """Return whether indices i and j are horizontally visible (i < j)."""
    if i > j:
        i, j = j, i
    if i == j:
        return False
    floor = min(values[i], values[j])
    return all(values[k] < floor for k in range(i + 1, j))


@dataclass
class AppendResult:
    """Result of appending one point to an incremental HVG."""

    index: int
    value: float
    new_edges: list[tuple]
    n_nodes: int
    n_edges: int


@dataclass
class IncrementalHVG:
    """
    Incrementally extend an HVG as new observations arrive.

    Appending a point at the end only creates edges involving the new index;
    existing edges among earlier nodes are unchanged.

    Parameters
    ----------
    weighted : bool, default False
        Weight edges by absolute value difference.
    limit : int, optional
        Maximum temporal distance between connected nodes.
    directed : bool, default False
        If True, edges point forward in time (i -> j, i < j).
    """

    weighted: bool = False
    limit: int | None = None
    directed: bool = False
    _values: list[float] = field(default_factory=list)
    _edges: list[tuple] = field(default_factory=list)

    @classmethod
    def from_series(
        cls,
        x: NDArray[np.float64],
        **kwargs,
    ) -> IncrementalHVG:
        """Build incrementally from an existing series (equivalent to batch)."""
        inc = cls(**kwargs)
        for v in np.asarray(x, dtype=np.float64).ravel():
            inc.append(float(v))
        return inc

    def append(self, value: float) -> AppendResult:
        """
        Append one observation and add any new visibility edges.

        Returns
        -------
        AppendResult
            New node index, edges born on this step, and totals.
        """
        j = len(self._values)
        self._values.append(float(value))
        new_edges: list[tuple] = []

        for i in range(j):
            if self.limit is not None and abs(j - i) > self.limit:
                continue
            if _hvg_visible(self._values, i, j):
                if self.weighted:
                    w = abs(self._values[i] - self._values[j])
                    edge: tuple = (i, j, w)
                else:
                    edge = (i, j)
                new_edges.append(edge)
                self._edges.append(edge)

        return AppendResult(
            index=j,
            value=float(value),
            new_edges=new_edges,
            n_nodes=len(self._values),
            n_edges=len(self._edges),
        )

    @property
    def n_nodes(self) -> int:
        return len(self._values)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def values(self) -> NDArray[np.float64]:
        return np.asarray(self._values, dtype=np.float64)

    def to_graph(self) -> Graph:
        """Current graph snapshot."""
        return Graph(
            edges=list(self._edges),
            n_nodes=len(self._values),
            directed=self.directed,
            weighted=self.weighted,
        )

    def stats(self) -> dict[str, float]:
        """Summary statistics for the current graph."""
        return self.to_graph().stats()
