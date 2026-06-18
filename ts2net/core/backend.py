"""
Unified compute backend selection for graph builders and distance kernels.

Backends: ``rust`` | ``numba`` | ``python`` (plus ``auto``).
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

ComputeBackend = Literal["auto", "rust", "numba", "python"]
_VALID = frozenset({"auto", "rust", "numba", "python"})


def rust_available() -> bool:
    """Return True when the ``ts2net_rs`` extension is importable."""
    try:
        import ts2net_rs  # noqa: F401

        return True
    except ImportError:
        return False


def numba_available() -> bool:
    """Return True when Numba is installed."""
    try:
        from numba import njit  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_compute_backend(
    requested: ComputeBackend | str = "auto",
    *,
    allow_fallback: bool = True,
) -> str:
    """
    Resolve a compute backend name to ``rust``, ``numba``, or ``python``.

    Parameters
    ----------
    requested : str, default ``auto``
        ``auto`` picks rust → numba → python. ``TS2NET_BACKEND`` overrides
        ``auto`` when set.
    allow_fallback : bool, default True
        When False, raise ``ImportError`` if the requested backend is missing.
    """
    req = (requested or "auto").lower()
    if req not in _VALID:
        raise ValueError(f"backend must be one of {sorted(_VALID)}, got {req!r}")

    if req == "auto":
        env = os.environ.get("TS2NET_BACKEND", "").lower()
        if env in ("rust", "numba", "python"):
            req = env

    if req == "auto":
        if rust_available():
            return "rust"
        if numba_available():
            return "numba"
        return "python"

    if req == "rust":
        if not rust_available():
            if allow_fallback:
                warnings.warn(
                    "Rust backend requested but ts2net_rs is unavailable; "
                    "falling back to numba/python.",
                    stacklevel=2,
                )
                return resolve_compute_backend("auto", allow_fallback=True)
            raise ImportError(
                "Rust backend requested but ts2net_rs is not built. "
                "Install with: uv sync --extra rust"
            )
        return "rust"

    if req == "numba":
        if not numba_available():
            if allow_fallback:
                warnings.warn(
                    "Numba backend requested but numba is unavailable; "
                    "using python.",
                    stacklevel=2,
                )
                return "python"
            raise ImportError("Numba backend requested but numba is not installed")
        return "numba"

    return "python"
