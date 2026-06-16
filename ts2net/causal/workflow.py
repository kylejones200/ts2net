"""
End-to-end causal network workflow.

Orchestrates lag search, confidence estimation, confounder adjustment,
network construction, and summary reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .confidence import te_bootstrap_ci, te_permutation_test
from .confounders import conditional_te_network, partial_granger_causality
from .granger import granger_causality
from .lag_search import search_granger_lag, search_te_lag
from .metrics import causal_network_metrics
from .summary import CausalAnalysisResult, CausalEdgeResult
from ._parallel import pairwise_parallel


@dataclass
class CausalWorkflowSpec:
    """Configuration for :func:`run_causal_analysis`."""

    method: Literal["granger", "transfer_entropy"] = "granger"
    max_lag: int = 5
    lag_search: bool = True
    te_lags: Optional[List[int]] = None
    alpha: float = 0.05
    adjust_confounders: bool = False
    n_permutations: int = 99
    bootstrap_ci: bool = False
    n_bootstrap: int = 100
    bins: int = 10
    te_threshold: Optional[float] = None
    granger_method: Literal["linear", "nonlinear"] = "linear"
    n_jobs: int = 1
    series_names: Optional[List[str]] = None


def run_causal_analysis(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    spec: Optional[CausalWorkflowSpec] = None,
    **kwargs,
) -> CausalAnalysisResult:
    """
    Run a full causal network workflow on a multivariate time series panel.

    Steps:
    1. Optional lag search per edge (Granger AIC/p-value or TE maximum)
    2. Pairwise causal tests with significance / permutation confidence
    3. Optional confounder adjustment (partial Granger or conditional TE)
    4. Network construction and topology metrics
    5. Plain-language summary via :meth:`CausalAnalysisResult.summary`

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Multivariate time series panel.
    spec : CausalWorkflowSpec, optional
        Workflow configuration. Extra ``**kwargs`` override spec fields.
    **kwargs
        Override any ``CausalWorkflowSpec`` field by name.

    Returns
    -------
    CausalAnalysisResult
        Graph, edge table, matrices, metrics, and report helpers.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x1 = rng.standard_normal(400)
    >>> x2 = np.concatenate([[0], 0.6 * x1[:-1] + 0.05 * rng.standard_normal(399)])
    >>> result = run_causal_analysis([x1, x2], method="granger", lag_search=True)
    >>> print(result.summary())
    """
    if spec is None:
        spec = CausalWorkflowSpec(**kwargs)
    else:
        for key, val in kwargs.items():
            if hasattr(spec, key):
                setattr(spec, key, val)

    X, names = _normalize_panel(X, spec.series_names)
    n_series = len(X)

    if spec.method == "granger":
        return _run_granger_workflow(X, names, spec)
    if spec.method == "transfer_entropy":
        return _run_te_workflow(X, names, spec)
    raise ValueError(
        f"Unknown method: {spec.method}. Use 'granger' or 'transfer_entropy'"
    )


def _run_granger_workflow(
    X: List[NDArray[np.float64]],
    names: List[str],
    spec: CausalWorkflowSpec,
) -> CausalAnalysisResult:
    n_series = len(X)
    lag_by_pair: Dict[Tuple[int, int], int] = {}
    edges: List[CausalEdgeResult] = []

    if spec.adjust_confounders:
        pair_results = _partial_granger_panel(X, spec, lag_by_pair)
    elif spec.lag_search:
        pair_results = _granger_with_lag_search(X, spec, lag_by_pair)
    else:
        pair_results = _granger_fixed_lag(X, spec)

    p_matrix = np.ones((n_series, n_series))
    G = nx.DiGraph()
    G.add_nodes_from(range(n_series))
    for i, name in enumerate(names):
        G.nodes[i]["name"] = name

    for (i, j), result in pair_results.items():
        p_val = result["p_value"]
        p_matrix[i, j] = p_val
        sig = p_val < spec.alpha
        weight = 1.0 - p_val if sig else 0.0
        best_lag = int(result.get("best_lag", spec.max_lag))

        edges.append(
            CausalEdgeResult(
                source=i,
                target=j,
                source_name=names[i],
                target_name=names[j],
                weight=weight,
                p_value=p_val,
                best_lag=best_lag,
                significant=sig,
                adjusted=spec.adjust_confounders,
            )
        )
        if sig:
            G.add_edge(
                i,
                j,
                weight=weight,
                p_value=p_val,
                best_lag=best_lag,
                adjusted=spec.adjust_confounders,
            )

    metrics = causal_network_metrics(G)
    metrics["alpha"] = spec.alpha
    metrics["method"] = "granger"
    metrics["lag_search"] = spec.lag_search

    return CausalAnalysisResult(
        graph=G,
        edges=edges,
        matrix=p_matrix,
        metrics=metrics,
        method="granger",
        lag_by_pair=lag_by_pair,
        alpha=spec.alpha,
        adjusted=spec.adjust_confounders,
        series_names=names,
    )


def _run_te_workflow(
    X: List[NDArray[np.float64]],
    names: List[str],
    spec: CausalWorkflowSpec,
) -> CausalAnalysisResult:
    n_series = len(X)
    lag_by_pair: Dict[Tuple[int, int], int] = {}
    edges: List[CausalEdgeResult] = []

    if spec.adjust_confounders:
        G, te_matrix, net_stats = conditional_te_network(
            X,
            lag=1,
            bins=spec.bins,
            threshold=spec.te_threshold,
            series_names=names,
            n_jobs=spec.n_jobs,
        )
        # Per-edge permutation tests on conditional TE
        for i, j in G.edges():
            lag = 1
            lag_by_pair[(i, j)] = lag
            others = [X[k] for k in range(n_series) if k not in (i, j)]
            z = np.mean(others, axis=0) if others else np.zeros_like(X[i])
            from .transfer_entropy import conditional_transfer_entropy

            te_val = conditional_transfer_entropy(X[i], X[j], z, lag=lag, bins=spec.bins)
            perm = te_permutation_test(
                X[i], X[j], lag=lag, bins=spec.bins,
                n_permutations=spec.n_permutations,
            )
            sig = perm["p_value"] < spec.alpha
            edges.append(
                CausalEdgeResult(
                    source=i, target=j,
                    source_name=names[i], target_name=names[j],
                    weight=te_val, p_value=perm["p_value"],
                    best_lag=lag, significant=sig, adjusted=True,
                )
            )
        for i in range(n_series):
            for j in range(n_series):
                if i != j and not any(e.source == i and e.target == j for e in edges):
                    edges.append(
                        CausalEdgeResult(
                            source=i, target=j,
                            source_name=names[i], target_name=names[j],
                            weight=float(te_matrix[i, j]),
                            p_value=1.0, best_lag=1,
                            significant=False, adjusted=True,
                        )
                    )
    else:
        def _analyze_pair(i: int, j: int) -> CausalEdgeResult:
            if spec.lag_search:
                best_lag, _ = search_te_lag(
                    X[i], X[j], lags=spec.te_lags, bins=spec.bins
                )
            else:
                best_lag = 1
            lag_by_pair[(i, j)] = best_lag

            perm = te_permutation_test(
                X[i], X[j], lag=best_lag, bins=spec.bins,
                n_permutations=spec.n_permutations,
            )
            ci_low = ci_high = None
            if spec.bootstrap_ci:
                ci = te_bootstrap_ci(
                    X[i], X[j], lag=best_lag, bins=spec.bins,
                    n_bootstrap=spec.n_bootstrap,
                )
                ci_low, ci_high = ci["ci_low"], ci["ci_high"]

            sig = perm["p_value"] < spec.alpha
            return CausalEdgeResult(
                source=i, target=j,
                source_name=names[i], target_name=names[j],
                weight=perm["te"], p_value=perm["p_value"],
                best_lag=best_lag, significant=sig,
                adjusted=False, ci_low=ci_low, ci_high=ci_high,
            )

        n_series = len(X)
        pair_edges = pairwise_parallel(n_series, _analyze_pair, n_jobs=spec.n_jobs)
        edges = list(pair_edges.values())

        G = nx.DiGraph()
        G.add_nodes_from(range(n_series))
        for i, name in enumerate(names):
            G.nodes[i]["name"] = name

        te_matrix = np.zeros((n_series, n_series))
        for edge in edges:
            te_matrix[edge.source, edge.target] = edge.weight
            if edge.significant:
                if spec.te_threshold is None or edge.weight >= spec.te_threshold:
                    G.add_edge(
                        edge.source, edge.target,
                        weight=edge.weight, p_value=edge.p_value,
                        best_lag=edge.best_lag,
                    )

    metrics = causal_network_metrics(G)
    metrics["alpha"] = spec.alpha
    metrics["method"] = "transfer_entropy"
    metrics["lag_search"] = spec.lag_search
    if not spec.adjust_confounders:
        te_matrix = np.zeros((n_series, n_series))
        for edge in edges:
            te_matrix[edge.source, edge.target] = edge.weight
    else:
        te_matrix = np.zeros((n_series, n_series))
        for edge in edges:
            te_matrix[edge.source, edge.target] = edge.weight

    return CausalAnalysisResult(
        graph=G,
        edges=edges,
        matrix=te_matrix,
        metrics=metrics,
        method="transfer_entropy",
        lag_by_pair=lag_by_pair,
        alpha=spec.alpha,
        adjusted=spec.adjust_confounders,
        series_names=names,
    )


def _granger_with_lag_search(
    X: List[NDArray[np.float64]],
    spec: CausalWorkflowSpec,
    lag_by_pair: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    n_series = len(X)

    def _test(i: int, j: int) -> Dict[str, float]:
        best_lag, scores = search_granger_lag(X[i], X[j], max_lag=spec.max_lag)
        lag_by_pair[(i, j)] = best_lag
        result = scores[best_lag]
        result["best_lag"] = float(best_lag)
        result["significant"] = result["p_value"] < spec.alpha
        return result

    return pairwise_parallel(n_series, _test, n_jobs=spec.n_jobs)


def _granger_fixed_lag(
    X: List[NDArray[np.float64]],
    spec: CausalWorkflowSpec,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    n_series = len(X)

    def _test(i: int, j: int) -> Dict[str, float]:
        return granger_causality(
            X[i], X[j], max_lag=spec.max_lag, method=spec.granger_method
        )

    return pairwise_parallel(n_series, _test, n_jobs=spec.n_jobs)


def _partial_granger_panel(
    X: List[NDArray[np.float64]],
    spec: CausalWorkflowSpec,
    lag_by_pair: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    n_series = len(X)

    def _test(i: int, j: int) -> Dict[str, float]:
        controls = [X[k] for k in range(n_series) if k not in (i, j)]
        if spec.lag_search:
            best_lag, _ = search_granger_lag(X[i], X[j], max_lag=spec.max_lag)
        else:
            best_lag = spec.max_lag
        lag_by_pair[(i, j)] = best_lag
        result = partial_granger_causality(
            X[i], X[j], controls, max_lag=best_lag
        )
        result["best_lag"] = float(best_lag)
        result["significant"] = result["p_value"] < spec.alpha
        return result

    return pairwise_parallel(n_series, _test, n_jobs=spec.n_jobs)


def _normalize_panel(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    series_names: Optional[List[str]],
) -> Tuple[List[NDArray[np.float64]], List[str]]:
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = [X]
        elif X.ndim == 2:
            X = [X[i] for i in range(X.shape[0])]
        else:
            raise ValueError(f"X must be 1D or 2D array, got shape {X.shape}")

    n_series = len(X)
    names = series_names or [f"Series_{i}" for i in range(n_series)]
    if len(names) != n_series:
        raise ValueError(
            f"series_names length ({len(names)}) must match "
            f"number of series ({n_series})"
        )
    return X, names
