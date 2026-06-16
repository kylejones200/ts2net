"""
Fast Causal Inference (FCI) for discovery with latent confounders.

Produces a Partial Ancestral Graph (PAG) with tail (-), arrowhead (>),
and circle (o) endpoint marks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .ci_tests import ci_test
from .lagged_panel import is_temporally_valid_edge, lagged_panel_matrix
from .pc import _learn_skeleton


TAIL = "tail"
ARROW = "arrow"
CIRCLE = "circle"


@dataclass
class FCIResult:
    """Output of the FCI algorithm."""

    pag: nx.DiGraph
    skeleton: nx.Graph
    separating_sets: Dict[Tuple[int, int], FrozenSet[int]]
    variable_names: List[str]
    alpha: float
    n_obs: int
    method: str = "fci"
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_networkx(self) -> nx.DiGraph:
        """Return the PAG with endpoint marks on edges."""
        return self.pag


def fci_algorithm(
    data: NDArray[np.float64],
    alpha: float = 0.05,
    variable_names: Optional[List[str]] = None,
    max_conditioning_set: Optional[int] = None,
    ci_method: Literal["partial_correlation"] = "partial_correlation",
) -> FCIResult:
    """
    Run the FCI algorithm with partial-correlation CI tests.

    FCI extends PC by representing possible latent confounders with
    circle marks and bidirected arrows in the output PAG.

    Parameters
    ----------
    data : array (n_samples, n_vars)
        Observations.
    alpha : float, default 0.05
        Significance level.
    variable_names : list of str, optional
        Variable labels.
    max_conditioning_set : int, optional
        Maximum conditioning set size.
    ci_method : {"partial_correlation"}
        Conditional independence test.

    Returns
    -------
    FCIResult
        Partial ancestral graph and separating sets.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> lat = rng.standard_normal(1000)
    >>> x = lat + 0.1 * rng.standard_normal(1000)
    >>> y = lat + 0.1 * rng.standard_normal(1000)
    >>> result = fci_algorithm(np.column_stack([x, y]), alpha=0.05)
    >>> result.pag.number_of_edges() >= 1
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
    marks = _initialize_circle_graph(n_vars, skeleton)
    marks = _fci_orient_colliders(marks, skeleton, sepsets)
    marks = _fci_propagation_rules(marks)
    pag = _marks_to_pag(marks, names)

    return FCIResult(
        pag=pag,
        skeleton=skeleton,
        separating_sets=sepsets,
        variable_names=names,
        alpha=alpha,
        n_obs=n_obs,
    )


def fci_timeseries_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    max_lag: int = 2,
    alpha: float = 0.05,
    series_names: Optional[List[str]] = None,
    allow_contemporaneous: bool = True,
    ci_method: Literal["partial_correlation"] = "partial_correlation",
) -> FCIResult:
    """
    FCI discovery on a lag-expanded multivariate time series panel.

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Multivariate panel.
    max_lag : int, default 2
        Maximum lag order.
    alpha : float, default 0.05
        Significance level.
    series_names : list of str, optional
        Series labels.
    allow_contemporaneous : bool, default True
        Allow contemporaneous edges.
    ci_method : {"partial_correlation"}
        CI test.

    Returns
    -------
    FCIResult
        PAG over lagged variables.
    """
    data, names, _ = lagged_panel_matrix(X, max_lag=max_lag, series_names=series_names)
    result = fci_algorithm(
        data,
        alpha=alpha,
        variable_names=names,
        ci_method=ci_method,
    )
    result = _filter_temporal_pag(result, allow_contemporaneous)
    result.metadata["max_lag"] = max_lag
    result.metadata["allow_contemporaneous"] = allow_contemporaneous
    result.method = "fci_timeseries"
    return result


def _initialize_circle_graph(
    n_vars: int,
    skeleton: nx.Graph,
) -> Dict[Tuple[int, int], str]:
    """Circle marks on both endpoints for each skeleton edge."""
    marks: Dict[Tuple[int, int], str] = {}
    for u, v in skeleton.edges():
        marks[(u, v)] = CIRCLE
        marks[(v, u)] = CIRCLE
    return marks


def _fci_orient_colliders(
    marks: Dict[Tuple[int, int], str],
    skeleton: nx.Graph,
    sepsets: Dict[Tuple[int, int], FrozenSet[int]],
) -> Dict[Tuple[int, int], str]:
    """R0: orient definite colliders a -> c <- b."""
    marks = dict(marks)
    for c in skeleton.nodes():
        nbrs = list(skeleton.neighbors(c))
        for a, b in combinations(nbrs, 2):
            if skeleton.has_edge(a, b):
                continue
            key = (min(a, b), max(a, b))
            sep = sepsets.get(key, frozenset())
            if c not in sep:
                marks[(a, c)] = ARROW
                marks[(c, a)] = TAIL
                marks[(b, c)] = ARROW
                marks[(c, b)] = TAIL
    return marks


def _fci_propagation_rules(
    marks: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], str]:
    """Apply FCI rules R1-R3 (circle removal and tail propagation)."""
    marks = dict(marks)
    nodes = _nodes_from_marks(marks)
    changed = True
    while changed:
        changed = False

        # R1: a o-> c, a and b adjacent, b o—c, b and c not adjacent => b o-> c
        for a in nodes:
            for c in nodes:
                if a == c or marks.get((a, c)) != ARROW:
                    continue
                if marks.get((c, a)) != CIRCLE:
                    continue
                for b in nodes:
                    if b in (a, c):
                        continue
                    if not _adjacent_marks(marks, a, b):
                        continue
                    if marks.get((b, c)) != CIRCLE or marks.get((c, b)) != CIRCLE:
                        continue
                    if _adjacent_marks(marks, b, c):
                        continue
                    marks[(b, c)] = ARROW
                    marks[(c, b)] = TAIL
                    changed = True

        # R2: a -> c o—b, a and b not adjacent => c o—b becomes c -> b
        for c in nodes:
            for b in nodes:
                if c == b:
                    continue
                if marks.get((c, b)) != CIRCLE or marks.get((b, c)) != CIRCLE:
                    continue
                for a in nodes:
                    if a in (b, c):
                        continue
                    if marks.get((a, c)) != ARROW or marks.get((c, a)) != TAIL:
                        continue
                    if _adjacent_marks(marks, a, b):
                        continue
                    marks[(c, b)] = ARROW
                    marks[(b, c)] = TAIL
                    changed = True

        # R3: a o—b o-> c, a and c not adjacent => a o-> c
        for b in nodes:
            for c in nodes:
                if b == c or marks.get((b, c)) != ARROW:
                    continue
                if marks.get((c, b)) != CIRCLE:
                    continue
                for a in nodes:
                    if a in (b, c):
                        continue
                    if marks.get((a, b)) != CIRCLE or marks.get((b, a)) != CIRCLE:
                        continue
                    if _adjacent_marks(marks, a, c):
                        continue
                    marks[(a, c)] = ARROW
                    marks[(c, a)] = TAIL
                    changed = True

        # R4: if a o—b and all marks at b are tails or arrows (no circle at b
        # toward any neighbor), convert remaining o— to directed tails/arrows
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                if marks.get((a, b)) != CIRCLE or marks.get((b, a)) != CIRCLE:
                    continue
                if _has_circle_at_node(marks, b):
                    continue
                marks[(a, b)] = TAIL
                marks[(b, a)] = ARROW
                changed = True

    return marks


def _nodes_from_marks(marks: Dict[Tuple[int, int], str]) -> Set[int]:
    nodes: Set[int] = set()
    for u, v in marks:
        nodes.add(u)
        nodes.add(v)
    return nodes


def _adjacent_marks(marks: Dict[Tuple[int, int], str], u: int, v: int) -> bool:
    return (u, v) in marks or (v, u) in marks


def _has_circle_at_node(marks: Dict[Tuple[int, int], str], node: int) -> bool:
    for (u, v), mark in marks.items():
        if u == node and mark == CIRCLE:
            return True
    return False


def _marks_to_pag(
    marks: Dict[Tuple[int, int], str],
    names: List[str],
) -> nx.DiGraph:
    """Convert endpoint marks to a directed graph with mark attributes."""
    pag = nx.DiGraph()
    for i, name in enumerate(names):
        pag.add_node(i, name=name)

    seen: Set[Tuple[int, int]] = set()
    for (u, v), mark_uv in marks.items():
        if (min(u, v), max(u, v)) in seen:
            continue
        if not marks.get((v, u)):
            continue
        seen.add((min(u, v), max(u, v)))
        pag.add_edge(
            u,
            v,
            mark_u=mark_uv,
            mark_v=marks[(v, u)],
            edge_label=_edge_label(mark_uv, marks[(v, u)]),
        )
    return pag


def _edge_label(mark_uv: str, mark_vu: str) -> str:
    """Human-readable PAG edge label from u's mark at v and v's mark at u."""
    sym = {TAIL: "-", ARROW: ">", CIRCLE: "o"}
    return f"{sym.get(mark_vu, '?')}{sym.get(mark_uv, '?')}"


def _filter_temporal_pag(
    result: FCIResult,
    allow_contemporaneous: bool,
) -> FCIResult:
    names = result.variable_names
    valid: Set[Tuple[int, int]] = set()
    for u, v in result.skeleton.edges():
        if is_temporally_valid_edge(
            names[u], names[v], allow_contemporaneous=allow_contemporaneous
        ):
            valid.add((min(u, v), max(u, v)))

    new_skeleton = nx.Graph()
    new_skeleton.add_nodes_from(result.skeleton.nodes(data=True))
    for u, v in valid:
        new_skeleton.add_edge(u, v)

    new_pag = nx.DiGraph()
    new_pag.add_nodes_from(result.pag.nodes(data=True))
    for u, v, data in result.pag.edges(data=True):
        if (min(u, v), max(u, v)) in valid:
            new_pag.add_edge(u, v, **data)

    new_sepsets = {
        k: v for k, v in result.separating_sets.items() if k in valid
    }

    return FCIResult(
        pag=new_pag,
        skeleton=new_skeleton,
        separating_sets=new_sepsets,
        variable_names=result.variable_names,
        alpha=result.alpha,
        n_obs=result.n_obs,
        method=result.method,
        metadata=dict(result.metadata),
    )
