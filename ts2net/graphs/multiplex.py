"""
Multiplex (multi-layer) graph representation.

Combines multiple edge types (e.g. correlation + recurrence + visibility)
into a single multi-layer structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

MergeStrategy = Literal["union", "intersection", "sum"]


@dataclass
class MultiplexGraph:
    """
      Container for named graph layers.

      Attributes
      ----------
      layers : dict[str, nx.Graph]
          Named edge layers.
    n_nodes : int
          Number of nodes (must be consistent across layers).
    """

    layers: dict[str, nx.Graph] = field(default_factory=dict)
    n_nodes: int = 0

    def add_layer(self, name: str, G: nx.Graph) -> None:
        """Add or replace a layer, aligning node sets across layers."""
        G = G.copy()
        if G.number_of_nodes() > 0:
            new_n = max(G.nodes()) + 1
            if new_n > self.n_nodes:
                self.n_nodes = new_n
                for layer in self.layers.values():
                    layer.add_nodes_from(range(self.n_nodes))
        G.add_nodes_from(range(self.n_nodes))
        self.layers[name] = G

    def layer_names(self) -> list[str]:
        return list(self.layers.keys())

    def aggregate_adjacency(
        self,
        strategy: MergeStrategy = "union",
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Merge layer adjacencies into a single matrix.

        ``union`` takes max weight per edge; ``intersection`` requires all
        layers; ``sum`` adds weighted adjacencies.
        """
        if not self.layers:
            return np.zeros((self.n_nodes, self.n_nodes))

        names = self.layer_names()
        weights = weights or {name: 1.0 for name in names}

        mats = []
        nodelist = list(range(self.n_nodes))
        for name in names:
            A = nx.to_numpy_array(
                self.layers[name], nodelist=nodelist, dtype=np.float64
            )
            mats.append(weights.get(name, 1.0) * A)

        if strategy == "union":
            return np.maximum.reduce(mats)
        if strategy == "intersection":
            mask = np.ones_like(mats[0], dtype=bool)
            for M in mats:
                mask &= M > 0
            result = np.minimum.reduce(mats)
            return np.where(mask, result, 0.0)
        if strategy == "sum":
            return np.sum(mats, axis=0)
        raise ValueError(f"Unknown strategy: {strategy}")

    def edges_in_layers(self, u: int, v: int) -> set[str]:
        """Return layer names where edge (u, v) exists."""
        found: set[str] = set()
        for name, G in self.layers.items():
            if G.has_edge(u, v) or G.has_edge(v, u):
                found.add(name)
        return found


def multiplex_graph(layers: dict[str, nx.Graph]) -> MultiplexGraph:
    """Create a multiplex graph from a dictionary of layers."""
    mg = MultiplexGraph()
    for name, G in layers.items():
        mg.add_layer(name, G)
    return mg
