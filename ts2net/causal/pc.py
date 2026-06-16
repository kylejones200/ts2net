"""
Peter-Clark (PC) constraint-based causal discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .ci_tests import ci_test
from .lagged_panel import (
    is_temporally_valid_edge,
    lagged_panel_matrix,
)


@dataclass
class PCResult:
    """Output of the PC algorithm."""

    cpdag: nx.DiGraph
    skeleton: nx.Graph
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]]
    variable_names: List[str]
    alpha: float
    n_obs: int
    method: str = "pc"
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_networkx(self) -> nx.DiGraph:
        """Return the oriented CPDAG."""
        return self.cpdag


def pc_algorithm(
    data: NDArray[np.float64],
    alpha: float = 0.05,
    variable_names: Optional[List[str]] = None,
    max_conditioning_set: Optional[int] = None,
    ci_method: Literal["partial_correlation"] = "partial_correlation",
) -> PCResult:
    """
    Run the PC algorithm with partial-correlation CI tests.

    Parameters
    ----------
    data : array (n_samples, n_vars)
        Observations (rows = samples).
    alpha : float, default 0.05
        Significance level for conditional independence tests.
    variable_names : list of str, optional
        Labels for each column.
    max_conditioning_set : int, optional
        Maximum conditioning set size (default: number of variables - 2).
    ci_method : {"partial_correlation"}, default "partial_correlation"
        Conditional independence test.

    Returns
    -------
    PCResult
        Skeleton, separating sets, and oriented CPDAG.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(800)
    >>> y = 0.8 * x + 0.1 * rng.standard_normal(800)
    >>> z = 0.8 * y + 0.1 * rng.standard_normal(800)
    >>> result = pc_algorithm(np.column_stack([x, y, z]), alpha=0.01)
    >>> result.cpdag.number_of_edges() >= 2
    True
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")

    n_obs, n_vars = data.shape
    if n_vars < 2:
        raise ValueError("data must have at least 2 variables")
    if n_obs < 10:
        raise ValueError(f"Need at least 10 observations, got {n_obs}")

    names = variable_names or [f"X{i}" for i in range(n_vars)]
    if len(names) != n_vars:
        raise ValueError(
            f"variable_names length ({len(names)}) must match n_vars ({n_vars})"
        )

    max_ord = (
        max_conditioning_set if max_conditioning_set is not None else n_vars - 2
    )
    max_ord = max(0, min(max_ord, n_vars - 2))

    skeleton, sepsets = _learn_skeleton(
        data, alpha=alpha, max_ord=max_ord, ci_method=ci_method
    )
    cpdag = _orient_pc(skeleton, sepsets, n_vars)

    for i, name in enumerate(names):
        cpdag.nodes[i]["name"] = name
        skeleton.nodes[i]["name"] = name

    return PCResult(
        cpdag=cpdag,
        skeleton=skeleton,
        separating_sets=sepsets,
        variable_names=names,
        alpha=alpha,
        n_obs=n_obs,
    )


def pc_timeseries_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    max_lag: int = 2,
    alpha: float = 0.05,
    series_names: Optional[List[str]] = None,
    allow_contemporaneous: bool = True,
    ci_method: Literal["partial_correlation"] = "partial_correlation",
) -> PCResult:
    """
    PC discovery on a lag-expanded multivariate time series panel.

    Time-aware filtering removes edges that violate temporal ordering
    (cause must not be more recent than effect).

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Multivariate panel.
    max_lag : int, default 2
        Maximum lag order for variable expansion.
    alpha : float, default 0.05
        Significance level.
    series_names : list of str, optional
        Series labels.
    allow_contemporaneous : bool, default True
        Allow edges among variables at the same time slice.
    ci_method : {"partial_correlation"}
        Conditional independence test.

    Returns
    -------
    PCResult
        CPDAG over lagged variables.
    """
    data, names, _ = lagged_panel_matrix(X, max_lag=max_lag, series_names=series_names)
    result = pc_algorithm(
        data,
        alpha=alpha,
        variable_names=names,
        ci_method=ci_method,
    )
    result = _filter_temporal_edges(result, allow_contemporaneous)
    result.metadata["max_lag"] = max_lag
    result.metadata["allow_contemporaneous"] = allow_contemporaneous
    result.method = "pc_timeseries"
    return result


def _learn_skeleton(
    data: NDArray[np.float64],
    alpha: float,
    max_ord: int,
    ci_method: str,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], FrozenSet[int]]]:
    n_vars = data.shape[1]
    adj: Dict[int, Set[int]] = {i: set(range(n_vars)) - {i} for i in range(n_vars)}
    sepsets: Dict[Tuple[int, int], FrozenSet[int]] = {}

    for order in range(max_ord + 1):
        removed_any = False
        for x in range(n_vars):
            neighbors = sorted(adj[x])
            for y in neighbors:
                if y not in adj[x]:
                    continue
                nbrs = sorted(adj[x] - {y})
                if len(nbrs) < order:
                    continue
                for cond in combinations(nbrs, order):
                    indep, _, _ = ci_test(
                        data, x, y, cond, alpha=alpha, method=ci_method  # type: ignore[arg-type]
                    )
                    if indep:
                        adj[x].discard(y)
                        adj[y].discard(x)
                        key = (min(x, y), max(x, y))
                        sepsets[key] = frozenset(cond)
                        removed_any = True
                        break
                if y not in adj[x]:
                    break
        if not removed_any:
            break

    skeleton = nx.Graph()
    skeleton.add_nodes_from(range(n_vars))
    for x in range(n_vars):
        for y in adj[x]:
            if x < y:
                skeleton.add_edge(x, y)
    return skeleton, sepsets


def _orient_pc(
    skeleton: nx.Graph,
    sepsets: Dict[Tuple[int, int], FrozenSet[int]],
    n_vars: int,
) -> nx.DiGraph:
    """Orient v-structures and apply Meek rules."""
    # Undirected edges in a DiGraph (both directions, marked undirected)
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(range(n_vars))
    for u, v in skeleton.edges():
        cpdag.add_edge(u, v, edge_type="undirected")
        cpdag.add_edge(v, u, edge_type="undirected")

    # Rule 0: v-structures
    for b in skeleton.nodes():
        nbrs = list(skeleton.neighbors(b))
        for a, c in combinations(nbrs, 2):
            if skeleton.has_edge(a, c):
                continue
            key = (min(a, c), max(a, c))
            sep = sepsets.get(key, frozenset())
            if b not in sep:
                _orient_collider(cpdag, a, b, c)

    _meek_rules(cpdag)
    return cpdag


def _orient_collider(
    cpdag: nx.DiGraph, a: int, b: int, c: int
) -> None:
    """Orient a -> b <- c."""
    _set_directed(cpdag, a, b)
    _set_directed(cpdag, c, b)


def _set_directed(cpdag: nx.DiGraph, u: int, v: int) -> None:
    """Set u -> v, removing reverse undirected arc if present."""
    if cpdag.has_edge(v, u):
        cpdag.remove_edge(v, u)
    cpdag.add_edge(u, v, edge_type="directed")


def _is_undirected(cpdag: nx.DiGraph, u: int, v: int) -> bool:
    return (
        cpdag.has_edge(u, v)
        and cpdag.has_edge(v, u)
        and cpdag[u][v].get("edge_type") == "undirected"
    )


def _meek_rules(cpdag: nx.DiGraph) -> None:
    """Apply Meek orientation rules until fixed point."""
    changed = True
    while changed:
        changed = False
        nodes = list(cpdag.nodes())

        # Rule 1: a — b -> c, a and c not adjacent => a -> b
        for a in nodes:
            for b in list(cpdag.predecessors(a)):
                if not _is_undirected(cpdag, a, b):
                    continue
                for c in list(cpdag.successors(b)):
                    if c == a:
                        continue
                    if _edge_type(cpdag, b, c) != "directed":
                        continue
                    if cpdag.has_edge(a, c) or cpdag.has_edge(c, a):
                        continue
                    _set_directed(cpdag, a, b)
                    changed = True

        # Rule 2: a -> b — c, a and c not adjacent => b -> c
        for b in nodes:
            for a in list(cpdag.predecessors(b)):
                if _edge_type(cpdag, a, b) != "directed":
                    continue
                for c in list(cpdag.successors(b)):
                    if c == a:
                        continue
                    if not _is_undirected(cpdag, b, c):
                        continue
                    if cpdag.has_edge(a, c) or cpdag.has_edge(c, a):
                        continue
                    _set_directed(cpdag, b, c)
                    changed = True

        # Rule 3: a — b — c, a -> c, b and c not adjacent => b -> c
        for b in nodes:
            for a in list(cpdag.predecessors(b)):
                if not _is_undirected(cpdag, a, b):
                    continue
                for c in list(cpdag.successors(b)):
                    if c == a:
                        continue
                    if not _is_undirected(cpdag, b, c):
                        continue
                    if _edge_type(cpdag, a, c) != "directed":
                        continue
                    if cpdag.has_edge(a, c) and cpdag.has_edge(c, a):
                        continue
                    _set_directed(cpdag, b, c)
                    changed = True


def _edge_type(cpdag: nx.DiGraph, u: int, v: int) -> str:
    if not cpdag.has_edge(u, v):
        return "none"
    return str(cpdag[u][v].get("edge_type", "directed"))


def _filter_temporal_edges(
    result: PCResult,
    allow_contemporaneous: bool,
) -> PCResult:
    """Remove edges that violate temporal ordering."""
    names = result.variable_names
    valid_pairs: Set[Tuple[int, int]] = set()
    for u, v in result.skeleton.edges():
        if is_temporally_valid_edge(
            names[u], names[v], allow_contemporaneous=allow_contemporaneous
        ):
            valid_pairs.add((min(u, v), max(u, v)))

    new_skeleton = nx.Graph()
    new_skeleton.add_nodes_from(result.skeleton.nodes(data=True))
    for u, v in valid_pairs:
        new_skeleton.add_edge(u, v)

    new_cpdag = nx.DiGraph()
    new_cpdag.add_nodes_from(result.cpdag.nodes(data=True))
    for u, v, data in result.cpdag.edges(data=True):
        if (min(u, v), max(u, v)) in valid_pairs:
            new_cpdag.add_edge(u, v, **data)

    new_sepsets = {
        key: val
        for key, val in result.separating_sets.items()
        if key in valid_pairs
    }

    return PCResult(
        cpdag=new_cpdag,
        skeleton=new_skeleton,
        separating_sets=new_sepsets,
        variable_names=result.variable_names,
        alpha=result.alpha,
        n_obs=result.n_obs,
        method=result.method,
        metadata=dict(result.metadata),
    )
