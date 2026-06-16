"""
Conditional independence tests for constraint-based causal discovery.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def partial_correlation_ci_test(
    data: NDArray[np.float64],
    x: int,
    y: int,
    cond: Sequence[int] = (),
    alpha: float = 0.05,
) -> Tuple[bool, float, float]:
    """
    Test whether variables ``x`` and ``y`` are conditionally independent given ``cond``.

    Uses a Fisher z-transform on the partial correlation (Gaussian assumption).

    Parameters
    ----------
    data : array (n_samples, n_vars)
        Observations.
    x, y : int
        Column indices to test.
    cond : sequence of int, default ()
        Conditioning variable indices.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    independent : bool
        True when the null (conditional independence) is not rejected.
    p_value : float
        Two-sided p-value.
    partial_r : float
        Estimated partial correlation.
    """
    cols = [x, y, *cond]
    sub = np.asarray(data[:, cols], dtype=np.float64)
    n, k = sub.shape
    if n < k + 2:
        return True, 1.0, 0.0

    sub = sub - sub.mean(axis=0, keepdims=True)
    std = sub.std(axis=0, ddof=1, keepdims=True)
    std[std < 1e-12] = 1.0
    sub = sub / std

    cov = (sub.T @ sub) / (n - 1)
    prec = np.linalg.pinv(cov)
    d = np.diag(prec)
    denom = np.sqrt(max(d[0] * d[1], 1e-12))
    partial_r = float(-prec[0, 1] / denom)
    partial_r = max(min(partial_r, 0.999999), -0.999999)

    df = n - k
    if df < 4:
        return True, 1.0, partial_r

    if abs(partial_r) >= 1.0:
        return False, 0.0, partial_r

    z = 0.5 * np.log((1 + partial_r) / (1 - partial_r))
    se = 1.0 / np.sqrt(df)
    z_stat = abs(z) / se
    p_value = float(2 * stats.norm.sf(z_stat))
    independent = p_value >= alpha
    return independent, p_value, partial_r


def ci_test(
    data: NDArray[np.float64],
    x: int,
    y: int,
    cond: Sequence[int] = (),
    alpha: float = 0.05,
    method: Literal["partial_correlation"] = "partial_correlation",
) -> Tuple[bool, float, float]:
    """
    Dispatch conditional independence tests.

    Parameters
    ----------
    data : array (n_samples, n_vars)
        Observations.
    x, y : int
        Variables to test.
    cond : sequence of int
        Conditioning set.
    alpha : float
        Significance level.
    method : {"partial_correlation"}, default "partial_correlation"
        Test to use.

    Returns
    -------
    independent, p_value, statistic
    """
    if method == "partial_correlation":
        return partial_correlation_ci_test(data, x, y, cond, alpha=alpha)
    raise ValueError(f"Unknown CI test method: {method}")
