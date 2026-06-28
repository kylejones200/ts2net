"""
Convert SINDy models to NetworkX coupling graphs.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from .core import SINDyResult


def sindy_coupling_network(
    result: SINDyResult,
    *,
    threshold: float = 0.05,
    linear_only: bool = True,
    include_self: bool = False,
) -> nx.DiGraph:
    """
    Build a directed coupling graph from SINDy coefficients.

    Nodes are state variables. An edge ``j → i`` is added when the discovered
    equation for ``state_i`` has a term involving ``state_j`` above ``threshold``.

    For a first-order polynomial library, off-diagonal linear terms correspond
    to direct couplings in ``ẋ = Ξ Θ(x)``.

    Parameters
    ----------
    result : SINDyResult
        Output of :func:`fit_sindy`.
    threshold : float
        Minimum |coefficient| to keep an edge.
    linear_only : bool, default True
        If True, only use pure state features (ignore ``x*y``, ``x^2``, etc.).
    include_self : bool, default False
        Include diagonal self-dynamics as self-loops.

    Returns
    -------
    networkx.DiGraph
        Edge attribute ``weight`` holds the SINDy coefficient.
    """
    G = nx.DiGraph()
    G.add_nodes_from(result.state_names)

    feature_to_state = {name: name for name in result.state_names}
    coef = result.coefficients

    for i, target in enumerate(result.state_names):
        for j, feat in enumerate(result.feature_names):
            weight = float(coef[i, j])
            if abs(weight) < threshold:
                continue

            if feat == "1":
                continue

            if linear_only and feat not in feature_to_state:
                continue

            source = feature_to_state.get(feat, feat)
            if source not in result.state_names:
                continue
            if source == target and not include_self:
                continue

            G.add_edge(source, target, weight=weight)

    G.graph["method"] = "sindy_coupling"
    G.graph["threshold"] = threshold
    G.graph["linear_only"] = linear_only
    return G


def sindy_jacobian_network(
    result: SINDyResult,
    x_eq: np.ndarray | None = None,
    *,
    threshold: float = 0.05,
) -> nx.DiGraph:
    """
    Linearization graph: evaluate Jacobian of the discovered field at ``x_eq``.

    Defaults to the origin when ``x_eq`` is None (appropriate for polynomial
    models when linear couplings dominate near zero).
    """
    n = len(result.state_names)
    x_eq = np.zeros(n) if x_eq is None else np.asarray(x_eq, dtype=np.float64).ravel()
    if len(x_eq) != n:
        raise ValueError(f"x_eq length {len(x_eq)} != n_states {n}")

    eps = 1e-6
    jac = np.zeros((n, n), dtype=np.float64)
    f0 = np.asarray(result.model.predict(x_eq.reshape(1, -1))[0], dtype=np.float64)
    for j in range(n):
        x_pert = x_eq.copy()
        x_pert[j] += eps
        fj = np.asarray(result.model.predict(x_pert.reshape(1, -1))[0], dtype=np.float64)
        jac[:, j] = (fj - f0) / eps
    G = nx.DiGraph()
    G.add_nodes_from(result.state_names)
    for i, target in enumerate(result.state_names):
        for j, source in enumerate(result.state_names):
            w = float(jac[i, j])
            if abs(w) >= threshold:
                G.add_edge(source, target, weight=w)
    G.graph["method"] = "sindy_jacobian"
    G.graph["x_eq"] = x_eq.tolist()
    return G
