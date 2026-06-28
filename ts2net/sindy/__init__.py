"""Sparse identification of nonlinear dynamics (PySINDy integration)."""

from ts2net.sindy.core import SINDyResult, SINDySpec, fit_sindy
from ts2net.sindy.network import sindy_coupling_network, sindy_jacobian_network

__all__ = [
    "SINDySpec",
    "SINDyResult",
    "fit_sindy",
    "sindy_coupling_network",
    "sindy_jacobian_network",
]
