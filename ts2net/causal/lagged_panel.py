"""
Lagged variable expansion for time-series causal discovery.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def lagged_panel_matrix(
    X: Union[List[NDArray[np.float64]], NDArray[np.float64]],
    max_lag: int = 2,
    series_names: Optional[List[str]] = None,
) -> Tuple[NDArray[np.float64], List[str], Dict[str, int]]:
    """
    Expand a multivariate panel into a lagged design matrix for PC/FCI.

    Each series contributes ``max_lag + 1`` columns:
    ``name(t)``, ``name(t-1)``, ..., ``name(t-max_lag)``.

    The first ``max_lag`` observations are dropped.

    Parameters
    ----------
    X : list of 1D arrays or array (n_series, n_points)
        Multivariate time series panel.
    max_lag : int, default 2
        Maximum lag order (must be >= 0).
    series_names : list of str, optional
        Names for each series.

    Returns
    -------
    data : array (n_obs, n_vars)
        Lag-expanded observations.
    names : list of str
        Column names.
    name_to_col : dict
        Map from variable name to column index.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = [X]
        elif X.ndim == 2:
            X = [X[i] for i in range(X.shape[0])]
        else:
            raise ValueError(f"X must be 1D or 2D array, got shape {X.shape}")

    n_series = len(X)
    if n_series == 0:
        raise ValueError("X must contain at least one series")

    lengths = {len(s) for s in X}
    if len(lengths) != 1:
        raise ValueError(f"All series must have the same length, got {lengths}")
    n_points = lengths.pop()
    if n_points <= max_lag:
        raise ValueError(
            f"Series length ({n_points}) must exceed max_lag ({max_lag})"
        )

    names_in = series_names or [f"series_{i}" for i in range(n_series)]
    if len(names_in) != n_series:
        raise ValueError(
            f"series_names length ({len(names_in)}) must match "
            f"number of series ({n_series})"
        )

    col_names: List[str] = []
    blocks: List[NDArray[np.float64]] = []
    for i, s in enumerate(X):
        s = np.asarray(s, dtype=np.float64).ravel()
        for lag in range(max_lag + 1):
            if lag == 0:
                col_names.append(f"{names_in[i]}(t)")
                blocks.append(s[max_lag:])
            else:
                col_names.append(f"{names_in[i]}(t-{lag})")
                blocks.append(s[max_lag - lag : n_points - lag])

    data = np.column_stack(blocks)
    name_to_col = {name: idx for idx, name in enumerate(col_names)}
    return data, col_names, name_to_col


def is_temporally_valid_edge(
    source_name: str,
    target_name: str,
    allow_contemporaneous: bool = True,
) -> bool:
    """
    Return whether an edge respects time ordering.

    Parses names produced by :func:`lagged_panel_matrix`.
    """
    src_lag = _parse_lag(source_name)
    tgt_lag = _parse_lag(target_name)
    if src_lag is None or tgt_lag is None:
        return True
    if src_lag < tgt_lag:
        return False
    if src_lag == tgt_lag == 0:
        return allow_contemporaneous
    return True


def _parse_lag(name: str) -> Optional[int]:
    if name.endswith("(t)"):
        return 0
    if "(t-" in name and name.endswith(")"):
        try:
            return int(name[name.index("(t-") + 3 : -1])
        except ValueError:
            return None
    return None
