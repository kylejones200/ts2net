"""
Shared sklearn-style builder API helpers.

Provides fit/transform/fit_transform mixins and consistent not-built errors.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ._validation import validate_series
from .exceptions import NotBuiltError


def require_fitted(obj: object, builder_name: str) -> None:
    """Raise if fit() has not been called."""
    if not getattr(obj, "_fitted", False):
        raise NotBuiltError(f"{builder_name}: Must call fit() before transform().")


def require_built(obj: object, builder_name: str) -> None:
    """Raise if build() has not produced a graph."""
    if getattr(obj, "_graph", None) is None:
        raise NotBuiltError(
            f"{builder_name}: Network not built. "
            "Call build(x) or fit_transform(x) first."
        )


@runtime_checkable
class NetworkBuilder(Protocol):
    """Protocol implemented by all network graph builders."""

    def build(self, x: NDArray[np.float64]) -> NetworkBuilder: ...

    def fit(self, x: NDArray[np.float64]) -> NetworkBuilder: ...

    def transform(self) -> nx.Graph: ...

    def fit_transform(self, x: NDArray[np.float64]) -> nx.Graph: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_edges(self) -> int: ...

    def degree_sequence(self) -> NDArray[np.int64]: ...

    def stats(self, include_triangles: bool = False) -> dict: ...


class SklearnBuildMixin:
    """
    sklearn-compatible fit/transform for builders that expose build().

    Subclasses must set ``_builder_name`` and implement ``build()``, ``as_networkx()``.
    """

    _builder_name: str = "NetworkBuilder"
    _x: NDArray[np.float64] | None = None
    _fitted: bool = False

    def fit(self, x: NDArray[np.float64]) -> SklearnBuildMixin:
        """Store and validate input; does not build until transform()."""
        self._x = validate_series(x, self._builder_name)
        self._fitted = True
        return self

    def transform(self) -> nx.Graph:
        """Build the network from data stored by fit()."""
        require_fitted(self, self._builder_name)
        assert self._x is not None
        self.build(self._x)  # type: ignore[attr-defined]
        return self.as_networkx(force=True)  # type: ignore[attr-defined]

    def fit_transform(self, x: NDArray[np.float64]) -> nx.Graph:
        """Fit and transform in one step."""
        return self.fit(x).transform()

    def _ensure_built(self) -> None:
        require_built(self, self._builder_name)
