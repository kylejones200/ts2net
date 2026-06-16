"""
Confidence estimation for causal edges (permutation tests).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .transfer_entropy import transfer_entropy


def te_permutation_test(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lag: int = 1,
    bins: int = 10,
    n_permutations: int = 99,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Permutation test for transfer entropy significance.

    Shuffles the source series to build a null distribution of TE values.

    Parameters
    ----------
    x, y : array (n,)
        Source and target time series.
    lag : int, default 1
        Transfer entropy lag.
    bins : int, default 10
        Discretization bins.
    n_permutations : int, default 99
        Number of permutations.
    random_state : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``te``, ``p_value``, ``significant`` (p < 0.05), ``null_mean``, ``null_std``.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    observed = transfer_entropy(x, y, lag=lag, bins=bins, method="discrete")
    rng = np.random.default_rng(random_state)
    null = np.array(
        [
            transfer_entropy(rng.permutation(x), y, lag=lag, bins=bins, method="discrete")
            for _ in range(n_permutations)
        ]
    )
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))

    return {
        "te": float(observed),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
    }


def te_bootstrap_ci(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    lag: int = 1,
    bins: int = 10,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for transfer entropy.

    Parameters
    ----------
    x, y : array (n,)
        Source and target time series.
    lag : int, default 1
        Transfer entropy lag.
    bins : int, default 10
        Discretization bins.
    n_bootstrap : int, default 100
        Bootstrap resamples.
    alpha : float, default 0.05
        Significance level (interval is ``1 - alpha``).
    random_state : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``te``, ``ci_low``, ``ci_high``, ``alpha``.
    """
    n = len(x)
    if n < lag + 5:
        te = transfer_entropy(x, y, lag=lag, bins=bins)
        return {"te": float(te), "ci_low": float(te), "ci_high": float(te), "alpha": alpha}

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples.append(
            transfer_entropy(x[idx], y[idx], lag=lag, bins=bins, method="discrete")
        )

    samples_arr = np.array(samples)
    lo = float(np.percentile(samples_arr, 100 * alpha / 2))
    hi = float(np.percentile(samples_arr, 100 * (1 - alpha / 2)))
    observed = transfer_entropy(x, y, lag=lag, bins=bins, method="discrete")

    return {
        "te": float(observed),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
    }
