"""
Time-lagged causal network analysis.

Builds separate causal networks per lag (or a multiplex summary) using
transfer entropy or Granger causality.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .granger import granger_causality_network
from .transfer_entropy import transfer_entropy_network


def time_lagged_causality_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    lags: Optional[List[int]] = None,
    method: Literal["transfer_entropy", "granger"] = "transfer_entropy",
    combine: Literal["per_lag", "max", "mean"] = "per_lag",
    n_jobs: int = 1,
    **kwargs,
) -> Union[
    Dict[int, Tuple[nx.DiGraph, NDArray, Dict[str, float]]],
    Tuple[nx.DiGraph, NDArray, Dict[str, float]],
]:
    """
    Analyze causal structure at multiple time lags.

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Panel of time series.
    lags : list of int, optional
        Lags to evaluate (default ``[1, 2, 3, 5]``).
    method : {"transfer_entropy", "granger"}, default "transfer_entropy"
        Causality measure.
    combine : {"per_lag", "max", "mean"}, default "per_lag"
        ``per_lag`` returns a dict keyed by lag.
        ``max`` / ``mean`` aggregate the weight matrices across lags.
    n_jobs : int, default 1
        Parallel workers passed to the underlying network builder.
    **kwargs
        Extra arguments for ``transfer_entropy_network`` or
        ``granger_causality_network`` (e.g. ``bins``, ``alpha``, ``max_lag``).

    Returns
    -------
    per_lag : dict[int, (G, matrix, stats)]
        When ``combine="per_lag"``.
    G, matrix, stats : tuple
        Combined network when ``combine`` is ``max`` or ``mean``.

    Examples
    --------
    >>> import numpy as np
    >>> X = [np.random.randn(300) for _ in range(3)]
    >>> results = time_lagged_causality_network(X, lags=[1, 2], method="transfer_entropy")
    >>> 1 in results and 2 in results
    True
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    if not lags:
        raise ValueError("lags must be a non-empty list")

    per_lag: Dict[int, Tuple[nx.DiGraph, NDArray, Dict[str, float]]] = {}

    for lag in lags:
        if method == "transfer_entropy":
            G, matrix, stats = transfer_entropy_network(
                X, lag=lag, n_jobs=n_jobs, **kwargs
            )
        elif method == "granger":
            gc_kwargs = dict(kwargs)
            gc_kwargs.setdefault("max_lag", lag)
            G, matrix, stats = granger_causality_network(
                X, n_jobs=n_jobs, **gc_kwargs
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'transfer_entropy' or 'granger'"
            )

        stats = dict(stats)
        stats["lag"] = lag
        per_lag[lag] = (G, matrix, stats)

    if combine == "per_lag":
        return per_lag

    matrices = [per_lag[lag][1] for lag in lags]
    if combine == "max":
        combined = np.maximum.reduce(matrices)
    elif combine == "mean":
        combined = np.mean(matrices, axis=0)
    else:
        raise ValueError(
            f"Unknown combine: {combine}. Use 'per_lag', 'max', or 'mean'"
        )

    # Rebuild graph from combined matrix
    n_series = combined.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n_series))

    threshold = kwargs.get("threshold")
    alpha = kwargs.get("alpha", 0.05)

    for i in range(n_series):
        for j in range(n_series):
            if i == j:
                continue
            val = combined[i, j]
            if method == "granger":
                if val >= alpha:
                    continue
                weight = 1.0 - val
                G.add_edge(i, j, weight=weight, p_value=val)
            else:
                if threshold is not None and val < threshold:
                    continue
                G.add_edge(i, j, weight=val)

    stats = {
        "lags": list(lags),
        "combine": combine,
        "method": method,
        "n_edges": G.number_of_edges(),
        "mean_weight": float(np.mean(combined[combined > 0]))
        if np.any(combined > 0)
        else 0.0,
    }

    return G, combined, stats
