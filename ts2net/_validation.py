"""
Input validation helpers for public APIs.

Centralizes series cleaning, parameter checks, and consistent error messages.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from .exceptions import ValidationError

OutputMode = Literal["edges", "degrees", "stats"]
VALID_OUTPUT_MODES: tuple[OutputMode, ...] = ("edges", "degrees", "stats")

T = TypeVar("T")


def validate_output_mode(output: str, builder_name: str = "ts2net") -> OutputMode:
    """Validate builder output mode."""
    if output not in VALID_OUTPUT_MODES:
        raise ValidationError(
            f"{builder_name}: output must be one of {VALID_OUTPUT_MODES}, "
            f"got {output!r}"
        )
    return output  # type: ignore[return-value]


def validate_positive_int(
    name: str,
    value: int,
    *,
    builder_name: str = "ts2net",
    minimum: int = 1,
) -> int:
    """Validate a strictly positive integer parameter."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{builder_name}: {name} must be an integer, got {type(value).__name__}"
        )
    if value < minimum:
        raise ValidationError(
            f"{builder_name}: {name} must be >= {minimum}, got {value}"
        )
    return value


def validate_choice(
    name: str,
    value: str,
    choices: Collection[str],
    *,
    builder_name: str = "ts2net",
) -> str:
    """Validate a string parameter against allowed choices."""
    if value not in choices:
        allowed = ", ".join(repr(c) for c in choices)
        raise ValidationError(
            f"{builder_name}: {name} must be one of [{allowed}], got {value!r}"
        )
    return value


def validate_series(
    x: object,
    builder_name: str = "ts2net",
    *,
    min_length: int = 1,
    warn_degenerate: bool = True,
) -> NDArray[np.float64]:
    """
    Validate and clean a univariate time series.

    Parameters
    ----------
    x : array-like
        Input time series (may contain non-numeric values).
    builder_name : str
        Builder name for error messages.
    min_length : int
        Minimum required length after cleaning.
    warn_degenerate : bool
        Whether to warn on constant or very short series.

    Returns
    -------
    x : ndarray, dtype float64
        Clean 1-D numeric array.

    Raises
    ------
    ValidationError
        If input cannot be converted to a valid numeric 1-D array.
    """
    import pandas as pd

    if not isinstance(x, np.ndarray):
        arr = np.asarray(x)
    else:
        arr = x

    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{builder_name}: Input must be array-like")

    if arr.ndim != 1:
        raise ValidationError(
            f"{builder_name}: Input must be a 1-D array, got shape {arr.shape}"
        )

    if arr.dtype.kind in ("f", "i", "u"):
        arr = arr.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = arr[~np.isnan(arr)]
    else:
        s = pd.Series(arr)
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        s = s.dropna()
        arr = s.values.astype(np.float64)

    if len(arr) < min_length:
        raise ValidationError(
            f"{builder_name}: No valid numeric values in input series "
            f"(length {len(arr)} < {min_length}). "
            "Check for non-numeric data, infinities, or all-null values."
        )

    if warn_degenerate:
        _warn_degenerate_series(arr, builder_name)

    return arr


def _warn_degenerate_series(x: NDArray[np.float64], builder_name: str) -> None:
    """Emit warnings for common degenerate inputs."""
    if np.std(x) == 0:
        warnings.warn(
            f"{builder_name}: Constant series detected (std=0). "
            "Results may be degenerate.",
            UserWarning,
            stacklevel=3,
        )

    if len(x) < 3:
        warnings.warn(
            f"{builder_name}: Very short series (n={len(x)}). "
            "Network may be trivial or degenerate.",
            UserWarning,
            stacklevel=3,
        )

    if len(x) > 100_000:
        warnings.warn(
            f"{builder_name}: Very long series (n={len(x)}). "
            "Consider using limit/output='degrees' or windowing.",
            UserWarning,
            stacklevel=3,
        )

    if np.any(np.abs(x) > 1e10):
        warnings.warn(
            f"{builder_name}: Series contains very large values "
            f"(max={np.max(np.abs(x)):.2e}). Numerical issues are possible.",
            UserWarning,
            stacklevel=3,
        )


# Backward-compatible alias used by existing tests and internal code.
_validate_and_clean_series = validate_series
