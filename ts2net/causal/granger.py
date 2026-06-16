"""
Granger causality tests and network construction.

Linear Granger causality uses VAR/OLS F-tests (statsmodels).
Nonlinear Granger causality compares restricted vs unrestricted
neural-network predictors with a permutation-based significance test.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ._parallel import pairwise_parallel


def _require_statsmodels():
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as exc:
        raise ImportError(
            "Linear Granger causality requires statsmodels. "
            "Install with: pip install 'ts2net[bsts]'"
        ) from exc
    return grangercausalitytests


def _build_lag_matrix(series: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    """Return lagged design matrix with rows aligned to time t."""
    n = len(series)
    if n <= max_lag:
        return np.empty((0, max_lag))

    rows = []
    for lag in range(1, max_lag + 1):
        rows.append(series[max_lag - lag : n - lag])
    return np.column_stack(rows)


def _linear_granger(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int,
    test: str,
) -> Dict[str, float]:
    """Test whether x Granger-causes y using statsmodels."""
    grangercausalitytests = _require_statsmodels()

    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")

    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    n = len(y)
    if n <= max_lag + 1:
        return {"f_stat": 0.0, "p_value": 1.0, "best_lag": 0.0, "significant": False}

    # statsmodels tests whether the second column Granger-causes the first.
    data = np.column_stack([y, x])
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    best_lag = 1
    best_p = 1.0
    best_f = 0.0

    for lag in range(1, max_lag + 1):
        test_result = results[lag][0][test]
        f_stat, p_value = float(test_result[0]), float(test_result[1])
        if p_value < best_p:
            best_p = p_value
            best_f = f_stat
            best_lag = lag

    return {
        "f_stat": best_f,
        "p_value": best_p,
        "best_lag": float(best_lag),
        "significant": best_p < 0.05,
    }


def _nonlinear_granger(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int,
    n_permutations: int = 49,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Nonlinear Granger test via MLP prediction improvement.

  Compares MSE of predicting y_t from y lags alone vs y and x lags.
    Significance estimated by permuting x before fitting the unrestricted model.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")

    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    y_lags = _build_lag_matrix(y, max_lag)
    x_lags = _build_lag_matrix(x, max_lag)
    target = y[max_lag:]

    if len(target) < max_lag + 5:
        return {
            "f_stat": 0.0,
            "p_value": 1.0,
            "best_lag": float(max_lag),
            "significant": False,
            "mse_improvement": 0.0,
        }

    rng = np.random.default_rng(random_state)

    def _fit_mse(features: NDArray[np.float64]) -> float:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        model = MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
        )
        model.fit(X_scaled, target)
        pred = model.predict(X_scaled)
        return float(np.mean((target - pred) ** 2))

    mse_restricted = _fit_mse(y_lags)
    mse_unrestricted = _fit_mse(np.column_stack([y_lags, x_lags]))
    observed_improvement = mse_restricted - mse_unrestricted

    perm_improvements = []
    for _ in range(n_permutations):
        x_perm = rng.permutation(x)
        x_lags_perm = _build_lag_matrix(x_perm, max_lag)
        mse_perm = _fit_mse(np.column_stack([y_lags, x_lags_perm]))
        perm_improvements.append(mse_restricted - mse_perm)

    p_value = float(np.mean(np.array(perm_improvements) >= observed_improvement))
    p_value = max(p_value, 1.0 / (n_permutations + 1))

    f_stat = max(0.0, observed_improvement / (mse_restricted + 1e-12))

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "best_lag": float(max_lag),
        "significant": p_value < 0.05,
        "mse_improvement": float(observed_improvement),
    }


def granger_causality(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int = 5,
    method: Literal["linear", "nonlinear"] = "linear",
    test: str = "ssr_ftest",
    n_permutations: int = 49,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Test whether ``x`` Granger-causes ``y``.

    Parameters
    ----------
    x : array (n,)
        Candidate cause time series.
    y : array (n,)
        Effect time series.
    max_lag : int, default 5
        Maximum lag order for the autoregressive model.
    method : {"linear", "nonlinear"}, default "linear"
        ``linear`` uses OLS/VAR F-tests (requires statsmodels).
        ``nonlinear`` uses MLP predictors with permutation testing.
    test : str, default "ssr_ftest"
        statsmodels test name (linear method only).
    n_permutations : int, default 49
        Permutation count for nonlinear significance.
    random_state : int, optional
        RNG seed for nonlinear method.

    Returns
    -------
    dict
        Keys: ``f_stat``, ``p_value``, ``best_lag``, ``significant``.
        Nonlinear method also returns ``mse_improvement``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(500)
    >>> y = np.concatenate([[0], 0.6 * x[:-1] + 0.1 * np.random.randn(499)])
    >>> result = granger_causality(x, y, max_lag=3)
    >>> result["p_value"] < 0.05
    True
    """
    if method == "linear":
        return _linear_granger(x, y, max_lag=max_lag, test=test)
    if method == "nonlinear":
        return _nonlinear_granger(
            x,
            y,
            max_lag=max_lag,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    raise ValueError(f"Unknown method: {method}. Use 'linear' or 'nonlinear'")


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


def granger_causality_network(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    max_lag: int = 5,
    method: Literal["linear", "nonlinear"] = "linear",
    alpha: float = 0.05,
    weight_by: Literal["p_value", "f_stat", "significance"] = "significance",
    series_names: Optional[List[str]] = None,
    n_jobs: int = 1,
    test: str = "ssr_ftest",
    n_permutations: int = 49,
    random_state: Optional[int] = None,
) -> Tuple[nx.DiGraph, NDArray, Dict[str, float]]:
    """
    Build a directed Granger-causality network over multiple time series.

    Edge ``(i, j)`` means series ``i`` Granger-causes series ``j``.

    Parameters
    ----------
    X : list of arrays or array (n_series, n_points)
        Panel of time series.
    max_lag : int, default 5
        Maximum autoregressive lag.
    method : {"linear", "nonlinear"}, default "linear"
        Granger test variant.
    alpha : float, default 0.05
        Significance threshold for retaining edges.
    weight_by : str, default "significance"
        Edge weight scheme: ``p_value`` (1 - p), ``f_stat``, or
        ``significance`` (1 if p < alpha else 0).
    series_names : list of str, optional
        Node labels.
    n_jobs : int, default 1
        Parallel workers for pairwise tests (-1 = all CPUs).
    test : str, default "ssr_ftest"
        statsmodels test (linear method only).
    n_permutations : int, default 49
        Permutations for nonlinear method.
    random_state : int, optional
        RNG seed for nonlinear method.

    Returns
    -------
    G : networkx.DiGraph
        Directed Granger network.
    p_matrix : array (n_series, n_series)
        p-values (p[i, j] = test of i → j).
    stats : dict
        Network summary statistics.

    Examples
    --------
    >>> import numpy as np
    >>> x1 = np.random.randn(400)
    >>> x2 = np.concatenate([[0], 0.5 * x1[:-1] + 0.1 * np.random.randn(399)])
    >>> G, p_mat, stats = granger_causality_network([x1, x2], max_lag=3)
    >>> G.has_edge(0, 1)
    True
    """
    X, names = _normalize_panel(X, series_names)
    n_series = len(X)

    def _test_pair(i: int, j: int) -> Dict[str, float]:
        return granger_causality(
            X[i],
            X[j],
            max_lag=max_lag,
            method=method,
            test=test,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    pair_results = pairwise_parallel(n_series, _test_pair, n_jobs=n_jobs)

    p_matrix = np.ones((n_series, n_series))
    f_matrix = np.zeros((n_series, n_series))

    for (i, j), result in pair_results.items():
        p_matrix[i, j] = result["p_value"]
        f_matrix[i, j] = result["f_stat"]

    G = nx.DiGraph()
    G.add_nodes_from(range(n_series))

    for i, name in enumerate(names):
        G.nodes[i]["name"] = name

    for (i, j), result in pair_results.items():
        p_val = result["p_value"]
        if p_val >= alpha:
            continue

        if weight_by == "p_value":
            weight = 1.0 - p_val
        elif weight_by == "f_stat":
            weight = result["f_stat"]
        elif weight_by == "significance":
            weight = 1.0
        else:
            raise ValueError(
                f"Unknown weight_by: {weight_by}. "
                "Use 'p_value', 'f_stat', or 'significance'"
            )

        G.add_edge(i, j, weight=weight, p_value=p_val, f_stat=result["f_stat"])

    n_sig = int(np.sum(p_matrix < alpha)) - int(np.sum(np.diag(p_matrix) < alpha))
    stats = {
        "n_edges": G.number_of_edges(),
        "n_significant_pairs": n_sig,
        "mean_p_value": float(np.mean(p_matrix[p_matrix < 1.0])),
        "min_p_value": float(np.min(p_matrix[p_matrix < 1.0]))
        if np.any(p_matrix < 1.0)
        else 1.0,
        "density": G.number_of_edges() / (n_series * (n_series - 1))
        if n_series > 1
        else 0.0,
        "alpha": alpha,
        "method": method,
    }

    return G, p_matrix, stats
