"""Public exception types for ts2net."""

from __future__ import annotations


class Ts2NetError(Exception):
    """Base exception for ts2net errors."""


class ValidationError(Ts2NetError, ValueError):
    """Raised when input data or parameters fail validation."""


class NotBuiltError(Ts2NetError, ValueError):
    """Raised when a network builder method is called before build()/fit()."""
