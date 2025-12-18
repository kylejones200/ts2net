"""
symbols.py - Symbolization methods for time series analysis.

This module provides functions for converting time series data into symbolic representations,
including equal-width binning, equal-frequency binning, and ordinal pattern extraction.
These methods are useful for various time series analysis tasks such as complexity analysis
and feature extraction.
"""

from __future__ import annotations
import numpy as np
from typing import Literal


def equal_width(x: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    s = np.digitize(x, edges[1:-1])
    return s.astype(np.int64)


def equal_freq(x: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0, 1, bins + 1)
    edges = np.quantile(x, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    s = np.digitize(x, np.unique(edges)[1:-1])
    return s.astype(np.int64)


def ordinal_patterns(
    x: np.ndarray,
    order: int,
    delay: int = 1,
    tie_rule: Literal["drop", "jitter", "stable"] = "stable",
    jitter: float = 1e-9,
) -> np.ndarray:
    x = np.asarray(x, float)
    n = x.size
    L = n - delay * (order - 1)
    if order < 2:
        raise ValueError("order must be >= 2")
    if L <= 0:
        raise ValueError("Series too short")
    if tie_rule == "jitter":
        x = x + np.random.default_rng(3363).normal(0.0, jitter, size=n)
    pats = np.empty(L, dtype=np.int64)
    base = np.arange(order)
    for i in range(L):
        w = x[i : i + delay * order : delay]
        if tie_rule == "drop" and np.unique(w).size < order:
            pats[i] = -1
            continue
        if tie_rule == "stable":
            idx = np.lexsort((base, w))
        else:
            idx = np.argsort(w, kind="mergesort")
        # Lehmer code
        code = 0
        used = np.zeros(order, dtype=np.int64)
        for j in range(order):
            r = idx[j]
            code += (r - used[:r].sum()) * np.math.factorial(order - j - 1)
            used[r] = 1
        pats[i] = code
    return pats[pats >= 0]
