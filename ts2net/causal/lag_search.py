"""
Automatic lag selection for causal tests.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .granger import _build_lag_matrix, _require_statsmodels
from .transfer_entropy import transfer_entropy


def search_granger_lag(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int = 10,
    criterion: Literal["pvalue", "aic", "bic"] = "pvalue",
    test: str = "ssr_ftest",
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Select the best Granger lag order for ``x`` → ``y``.

    Parameters
    ----------
    x, y : array (n,)
        Cause and effect time series.
    max_lag : int, default 10
        Maximum lag to evaluate.
    criterion : {"pvalue", "aic", "bic"}, default "pvalue"
        ``pvalue`` picks the lag with the smallest p-value.
        ``aic`` / ``bic`` pick the lag with the lowest information criterion
        from the unrestricted AR model.
    test : str, default "ssr_ftest"
        statsmodels Granger test name.

    Returns
    -------
    best_lag : int
        Selected lag order.
    scores : dict
        Per-lag statistics (``p_value``, ``f_stat``, and optionally ``aic``/``bic``).
    """
    if len(x) != len(y):
        raise ValueError(f"Series must have same length: {len(x)} != {len(y)}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")

    grangercausalitytests = _require_statsmodels()
    data = np.column_stack([y, x])
    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    scores: Dict[int, Dict[str, float]] = {}
    for lag in range(1, max_lag + 1):
        test_result = results[lag][0][test]
        f_stat, p_value = float(test_result[0]), float(test_result[1])
        entry: Dict[str, float] = {"p_value": p_value, "f_stat": f_stat}

        if criterion in ("aic", "bic"):
            y_lags = _build_lag_matrix(y, lag)
            x_lags = _build_lag_matrix(x, lag)
            target = y[lag:]
            design = np.column_stack([y_lags, x_lags])
            n = len(target)
            k = design.shape[1] + 1  # intercept
            rss = _regression_rss(design, target)
            entry["aic"] = n * np.log(rss / n + 1e-12) + 2 * k
            entry["bic"] = n * np.log(rss / n + 1e-12) + k * np.log(n)

        scores[lag] = entry

    if criterion == "pvalue":
        best_lag = min(scores, key=lambda lag: scores[lag]["p_value"])
    else:
        best_lag = min(scores, key=lambda lag: scores[lag][criterion])

    return best_lag, scores


def search_te_lag(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lags: Optional[List[int]] = None,
    bins: int = 10,
    method: Literal["discrete", "knn"] = "discrete",
) -> Tuple[int, Dict[int, float]]:
    """
    Select the lag with maximum transfer entropy from ``x`` to ``y``.

    Parameters
    ----------
    x, y : array (n,)
        Source and target time series.
    lags : list of int, optional
        Candidate lags (default ``[1, 2, 3, 5, 10]`` capped by series length).
    bins : int, default 10
        Discretization bins for discrete TE.
    method : {"discrete", "knn"}, default "discrete"
        Transfer entropy estimator.

    Returns
    -------
    best_lag : int
        Lag with highest TE.
    scores : dict
        Transfer entropy at each evaluated lag.
    """
    if lags is None:
        max_eval = min(10, len(x) // 4)
        lags = [lag for lag in [1, 2, 3, 5, 10] if lag <= max_eval]
        if not lags:
            lags = [1]

    scores = {
        lag: transfer_entropy(x, y, lag=lag, bins=bins, method=method) for lag in lags
    }
    best_lag = max(scores, key=scores.get)
    return best_lag, scores


def _regression_rss(design: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    """Residual sum of squares for OLS with intercept."""
    x_ = np.column_stack([np.ones(len(target)), design])
    coef, _, _, _ = np.linalg.lstsq(x_, target, rcond=None)
    resid = target - x_ @ coef
    return float(np.sum(resid**2))
