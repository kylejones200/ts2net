"""The single owner of feature standardization for the role-feature pipeline.

Architecture::

    raw feature construction   (_role_features_basic, roles.py helpers)
            |
            v
    concatenation              (role_features_extended)
            |
            v
    ONE standardization boundary   <-- this module
            |
            v
    role feature matrix

No other module standardizes role features. Feature builders return raw
columns; whoever assembles the final matrix calls :func:`standardize` exactly
once.

Why this module exists
----------------------
The previous pipeline standardized twice: once inside ``_role_features_basic``
and again over the concatenated matrix. Each pass divided by ``std + 1e-12``.
For a column that is mathematically constant -- eigenvector centrality on any
vertex-transitive graph, k-core number on a Barabasi-Albert graph -- the raw
standard deviation is floating-point residue of order 1e-15. The first pass's
epsilon dominated that, leaving a column at roughly 1e-3; the second pass saw a
standard deviation far above the epsilon and renormalized the residue to unit
variance. Meaningless numerical noise became a full-amplitude feature, and
because ``eigenvector_centrality_numpy`` starts ARPACK from a random vector, the
same graph produced materially different feature matrices on repeated calls.

Two changes prevent that:

* standardization happens exactly once, here;
* a numerically degenerate column standardizes to exactly ``0.0`` rather than
  to rescaled residue.
"""

from __future__ import annotations

import numpy as np

#: Relative tolerance for calling a column numerically constant. A column
#: qualifies when its standard deviation is at most this fraction of its
#: magnitude. Chosen roughly three orders above the residue actually observed
#: in these features (relative spreads of 1e-16 to 1e-14 for mathematically
#: constant centralities) and many orders below any real variation: genuine
#: graph-feature signal differs in the second or third significant figure, not
#: the twelfth.
DEGENERACY_RTOL = 1e-12

#: Absolute floor, so a column whose values are numerical noise *around zero*
#: is caught too. Such a column has no magnitude for the relative test to work
#: against, so the relative criterion alone would classify pure noise as signal.
DEGENERACY_ATOL = 1e-12


def degeneracy_tolerance(
    x: np.ndarray,
    rtol: float = DEGENERACY_RTOL,
    atol: float = DEGENERACY_ATOL,
) -> np.ndarray:
    """Per-column threshold below which a standard deviation counts as noise.

    Scale aware: ``max(atol, rtol * scale)`` where ``scale`` is the larger of
    the column's mean magnitude and its largest absolute value. Using the
    maximum of the two keeps the test meaningful both for a constant far from
    zero (large ``|mean|``) and for values straddling zero (small ``|mean|``,
    non-trivial ``max|x|``).
    """
    mean_magnitude = np.abs(x.mean(axis=0))
    peak_magnitude = np.abs(x).max(axis=0) if x.shape[0] else np.zeros(x.shape[1])
    scale = np.maximum(mean_magnitude, peak_magnitude)
    return np.maximum(atol, rtol * scale)


def degenerate_columns(
    x: np.ndarray,
    rtol: float = DEGENERACY_RTOL,
    atol: float = DEGENERACY_ATOL,
) -> np.ndarray:
    """Boolean mask of columns that carry no resolvable variation.

    ``std == 0`` is deliberately *not* the criterion. A mathematically constant
    graph feature computed in floating point typically has a spread of order
    1e-15 rather than exactly zero, and treating that as signal is the defect
    this module exists to prevent.

    Fewer than two rows means variation is undefined, so every column is
    degenerate.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {x.shape}")
    if x.shape[0] < 2:
        return np.ones(x.shape[1], dtype=bool)
    return x.std(axis=0, ddof=1) <= degeneracy_tolerance(x, rtol, atol)


def standardize(
    x: np.ndarray,
    rtol: float = DEGENERACY_RTOL,
    atol: float = DEGENERACY_ATOL,
) -> np.ndarray:
    """Standardize each column to zero mean and unit variance, once.

    Degenerate columns (see :func:`degenerate_columns`) are returned as exactly
    ``0.0``. Every other column has a standard deviation strictly above the
    tolerance, so it is divided directly -- there is no ``+ epsilon`` guard,
    because the tolerance already established that the divisor is safe. An
    epsilon should protect the arithmetic, not convert numerical residue into
    apparent structure.

    Applying this twice is a no-op: the output of the first call is either
    exactly zero or has unit variance, and both are fixed points.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {x.shape}")

    out = np.zeros_like(x)
    if x.shape[0] < 2:
        return out

    keep = ~degenerate_columns(x, rtol, atol)
    if keep.any():
        varying = x[:, keep]
        out[:, keep] = (varying - varying.mean(axis=0)) / varying.std(axis=0, ddof=1)
    return out
