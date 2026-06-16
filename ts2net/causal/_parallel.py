"""Parallel pairwise computation helpers for causal network builders."""

from __future__ import annotations

from typing import Callable, Dict, Tuple, TypeVar

T = TypeVar("T")


def pairwise_parallel(
    n_series: int,
    compute: Callable[[int, int], T],
    n_jobs: int = 1,
) -> Dict[Tuple[int, int], T]:
    """
    Compute values for all ordered pairs (i, j) with i != j.

    Parameters
    ----------
    n_series : int
        Number of series (nodes).
    compute : callable
        ``compute(i, j)`` returns the value for pair (i, j).
    n_jobs : int, default 1
        Parallel workers. Use -1 for all CPUs.

    Returns
    -------
    dict
        Mapping (i, j) -> result.
    """
    pairs = [(i, j) for i in range(n_series) for j in range(n_series) if i != j]

    if n_jobs == 1 or len(pairs) <= 1:
        return {pair: compute(*pair) for pair in pairs}

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(compute)(i, j) for i, j in pairs
    )
    return dict(zip(pairs, results, strict=True))
