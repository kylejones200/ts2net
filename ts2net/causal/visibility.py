"""
Directional visibility graph analysis for temporal asymmetry.

Uses directed horizontal visibility graphs (DHVG) to quantify irreversibility
and time-arrow asymmetry in univariate time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ts2net.api import HVG
from ts2net._validation import validate_series


@dataclass
class VisibilityAsymmetryResult:
    """Directed visibility graph asymmetry metrics."""

    irreversibility_score: float
    temporal_asymmetry_index: float
    forward_backward_ratio: float
    graph: nx.DiGraph
    in_degrees: NDArray[np.float64]
    out_degrees: NDArray[np.float64]
    node_asymmetry: NDArray[np.float64]
    stats: Dict[str, float]
    method: str = "dhvg"
    metadata: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Plain-text summary of visibility asymmetry."""
        lines = [
            f"Directed visibility analysis ({self.method})",
            f"Irreversibility score: {self.irreversibility_score:.4f}",
            f"Temporal asymmetry index: {self.temporal_asymmetry_index:.4f}",
            f"Forward/backward ratio: {self.forward_backward_ratio:.4f}",
            f"Nodes: {self.stats.get('n_nodes', 0)}, "
            f"edges: {self.stats.get('n_edges', 0)}",
        ]
        return "\n".join(lines)


def directed_visibility_analysis(
    x: NDArray[np.float64],
    weighted: bool = False,
    limit: Optional[int] = None,
    compare_reversed: bool = True,
) -> VisibilityAsymmetryResult:
    """
    Analyze temporal asymmetry via a directed horizontal visibility graph.

    Directed HVG edges point forward in time (i → j for i < j), enabling
    irreversibility and time-arrow statistics useful for fault detection
    and causal asymmetry screening.

    Parameters
    ----------
    x : array (n,)
        Input time series.
    weighted : bool, default False
        Use absolute-difference edge weights.
    limit : int, optional
        Maximum temporal distance between connected nodes.
    compare_reversed : bool, default True
        When True, compute temporal asymmetry vs the time-reversed series.

    Returns
    -------
    VisibilityAsymmetryResult
        Irreversibility score, asymmetry index, graph, and degree sequences.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 200)
    >>> result = directed_visibility_analysis(x)
    >>> result.irreversibility_score >= 0
    True

    References
    ----------
    Lacasa et al. (2008). From time series to complex networks: The visibility
    graph. *PNAS*, 105(13), 4972–4975.
    """
    x = validate_series(x, "directed_visibility_analysis", warn_degenerate=False)
    hvg = HVG(directed=True, weighted=weighted, limit=limit)
    hvg.build(x)

    stats = hvg.stats()
    in_deg = hvg.in_degree_sequence().astype(np.float64)
    out_deg = hvg.out_degree_sequence().astype(np.float64)
    node_asym = _node_asymmetry(in_deg, out_deg)
    irrev = float(stats.get("irreversibility_score", 0.0))

    asym_index = 0.0
    fb_ratio = 1.0
    if compare_reversed and len(x) > 2:
        rev = directed_visibility_analysis(
            x[::-1].copy(),
            weighted=weighted,
            limit=limit,
            compare_reversed=False,
        )
        asym_index = irrev - rev.irreversibility_score
        fb_ratio = irrev / (rev.irreversibility_score + 1e-12)

    graph = hvg.as_networkx()
    return VisibilityAsymmetryResult(
        irreversibility_score=irrev,
        temporal_asymmetry_index=float(asym_index),
        forward_backward_ratio=float(fb_ratio),
        graph=graph,
        in_degrees=in_deg,
        out_degrees=out_deg,
        node_asymmetry=node_asym,
        stats={k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
        method="dhvg",
        metadata={"weighted": weighted, "limit": limit},
    )


def visibility_irreversibility(
    x: NDArray[np.float64],
    weighted: bool = False,
    limit: Optional[int] = None,
) -> float:
    """
    Scalar irreversibility score from a directed HVG.

    Parameters
    ----------
    x : array (n,)
        Input time series.
    weighted : bool, default False
        Use absolute-difference edge weights.
    limit : int, optional
        Maximum temporal distance between connected nodes.

    Returns
    -------
    float
        Mean node-level |in_degree - out_degree| / total_degree in [0, 1].
    """
    return directed_visibility_analysis(
        x, weighted=weighted, limit=limit, compare_reversed=False
    ).irreversibility_score


def visibility_asymmetry_panel(
    X: NDArray[np.float64],
    axis: int = 0,
    **kwargs,
) -> Dict[int, VisibilityAsymmetryResult]:
    """
    Run directed visibility analysis on each series in a panel.

    Parameters
    ----------
    X : array (n_series, n_points) or (n_points, n_series)
        Multivariate panel.
    axis : int, default 0
        Axis indexing individual series.
    **kwargs
        Passed to :func:`directed_visibility_analysis`.

    Returns
    -------
    dict[int, VisibilityAsymmetryResult]
        Results keyed by series index.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    results: Dict[int, VisibilityAsymmetryResult] = {}
    for i in range(X.shape[axis]):
        series = np.take(X, i, axis=axis)
        results[i] = directed_visibility_analysis(series, **kwargs)
    return results


def _node_asymmetry(
    in_deg: NDArray[np.float64],
    out_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    total = in_deg + out_deg
    asym = np.zeros_like(in_deg)
    mask = total > 0
    asym[mask] = np.abs(in_deg[mask] - out_deg[mask]) / total[mask]
    return asym
