"""
SINDy dynamics discovery — Rust core with optional PySINDy fallback.

Uses the native ``ts2net_rs`` STLSQ implementation when available; falls back
to PySINDy for advanced optimizers and simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from ..core.backend import rust_available

SindyBackend = Literal["auto", "rust", "pysindy"]


def _require_pysindy():
    try:
        import pysindy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PySINDy integration requires pysindy. "
            "Install with: pip install 'ts2net[sindy]'"
        ) from exc


@dataclass
class SINDySpec:
    """
    Configuration for a SINDy fit.

    Defaults follow PySINDy tutorial 1: finite-difference derivatives,
    polynomial library, and sequentially-thresholded least squares (STLSQ).
    """

    polynomial_degree: int = 3
    threshold: float = 0.1
    alpha: float = 0.05
    differentiation_order: int = 2
    optimizer: str = "stlsq"
    backend: SindyBackend = "auto"


@dataclass
class SINDyResult:
    """Outcome of :func:`fit_sindy`."""

    model: Any
    coefficients: NDArray[np.float64]
    feature_names: list[str]
    state_names: list[str]
    t: NDArray[np.float64] | float | list[NDArray[np.float64]] | None = None
    spec: SINDySpec = field(default_factory=SINDySpec)
    backend_used: str = "pysindy"

    def equations(self) -> list[str]:
        """Human-readable ODE right-hand sides (one per state)."""
        lines: list[str] = []
        coef = self.coefficients
        for i, state in enumerate(self.state_names):
            terms: list[str] = []
            for j, feat in enumerate(self.feature_names):
                c = coef[i, j]
                if abs(c) < 1e-8:
                    continue
                terms.append(f"{c:+.3f} {feat}")
            rhs = " ".join(terms) if terms else "0"
            lines.append(f"({state})' = {rhs}")
        return lines

    def simulate(
        self,
        x0: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Integrate the discovered model from ``x0`` over ``t``."""
        if self.model is None:
            raise NotImplementedError(
                "Simulation requires a PySINDy model. "
                "Re-fit with spec.backend='pysindy' or install pysindy and use "
                "fit_sindy(..., spec=SINDySpec(backend='pysindy'))."
            )
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        t = np.asarray(t, dtype=np.float64).ravel()
        return np.asarray(self.model.simulate(x0, t=t), dtype=np.float64)


def _normalize_X(
    X: NDArray[np.float64] | list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64] | list[NDArray[np.float64]], int]:
    if isinstance(X, list):
        if not X:
            raise ValueError("X must be a non-empty list of trajectories")
        n_vars = X[0].shape[-1]
        out: list[NDArray[np.float64]] = []
        for i, traj in enumerate(X):
            arr = np.asarray(traj, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValueError(f"Trajectory {i} must be 1D or 2D, got shape {arr.shape}")
            if arr.shape[-1] != n_vars:
                raise ValueError("All trajectories must have the same number of coordinates")
            out.append(arr)
        return out, n_vars

    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = validate_series(arr, "fit_sindy").reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(f"X must be (n_time, n_coords) or a list of arrays, got {arr.shape}")
    return arr, arr.shape[1]


def _resolve_sindy_backend(spec: SINDySpec) -> str:
    backend = (spec.backend or "auto").lower()
    if backend not in ("auto", "rust", "pysindy"):
        raise ValueError("backend must be 'auto', 'rust', or 'pysindy'")
    if backend == "auto":
        return "rust" if rust_available() else "pysindy"
    if backend == "rust" and not rust_available():
        raise ImportError(
            "Rust SINDy backend requested but ts2net_rs is not built. "
            "Install with: pip install ts2net[speed] or build from source."
        )
    return backend


def _fit_sindy_rust(
    X_norm: NDArray[np.float64] | list[NDArray[np.float64]],
    t: NDArray[np.float64] | float | list[NDArray[np.float64]],
    *,
    state_names: list[str],
    spec: SINDySpec,
) -> tuple[NDArray[np.float64], list[str]]:
    import ts2net_rs

    if spec.optimizer.lower() != "stlsq":
        raise ValueError(
            f"Rust SINDy backend supports optimizer='stlsq' only, got {spec.optimizer!r}"
        )

    kwargs = dict(
        state_names=state_names,
        polynomial_degree=int(spec.polynomial_degree),
        threshold=float(spec.threshold),
        alpha=float(spec.alpha),
        differentiation_order=int(spec.differentiation_order),
    )

    if isinstance(X_norm, list):
        if not isinstance(t, list):
            raise ValueError("t must be a list when X is a list of trajectories")
        coef, names = ts2net_rs.fit_sindy_rust_multi(
            [np.asarray(x, dtype=np.float64) for x in X_norm],
            [np.asarray(a, dtype=np.float64) for a in t],
            **kwargs,
        )
    else:
        if isinstance(t, list):
            raise ValueError("t must be a scalar or 1-D array for a single trajectory")
        t_arr = np.asarray(t, dtype=np.float64) if not isinstance(t, (float, int)) else None
        if isinstance(t, (float, int)):
            n = X_norm.shape[0]
            t_arr = np.linspace(0.0, float(t) * (n - 1), n, dtype=np.float64)
        coef, names = ts2net_rs.fit_sindy_rust(
            np.asarray(X_norm, dtype=np.float64),
            t_arr,
            **kwargs,
        )
    return np.asarray(coef, dtype=np.float64), list(names)


def _fit_sindy_pysindy(
    X_norm: NDArray[np.float64] | list[NDArray[np.float64]],
    t: NDArray[np.float64] | float | list[NDArray[np.float64]],
    *,
    x_dot: NDArray[np.float64] | list[NDArray[np.float64]] | None,
    state_names: list[str],
    spec: SINDySpec,
) -> tuple[Any, NDArray[np.float64], list[str]]:
    _require_pysindy()
    import pysindy as ps

    differentiation_method = ps.FiniteDifference(order=spec.differentiation_order)
    feature_library = ps.PolynomialLibrary(degree=spec.polynomial_degree)
    if spec.optimizer.lower() == "stlsq":
        optimizer = ps.STLSQ(threshold=spec.threshold, alpha=spec.alpha)
    else:
        raise ValueError(f"Unknown optimizer {spec.optimizer!r}. Use 'stlsq'.")

    model = ps.SINDy(
        differentiation_method=differentiation_method,
        feature_library=feature_library,
        optimizer=optimizer,
    )

    fit_kwargs: dict[str, Any] = {"feature_names": state_names}
    if x_dot is not None:
        fit_kwargs["x_dot"] = x_dot

    model.fit(X_norm, t=t, **fit_kwargs)
    coef = np.asarray(model.coefficients(), dtype=np.float64)
    lib_names = list(model.get_feature_names())
    return model, coef, lib_names


def fit_sindy(
    X: NDArray[np.float64] | list[NDArray[np.float64]],
    t: NDArray[np.float64] | float | list[NDArray[np.float64]],
    *,
    x_dot: NDArray[np.float64] | list[NDArray[np.float64]] | None = None,
    feature_names: list[str] | None = None,
    spec: SINDySpec | None = None,
) -> SINDyResult:
    """
    Fit a SINDy model to multivariate time-series data.

    Parameters
    ----------
    X : array (n_time, n_coords) or list of such arrays
        State observations. PySINDy axis convention: time first, coordinate second.
    t : array, scalar dt, or list of time arrays
        Sample times. Pass scalar ``dt`` when the timestep is uniform.
    x_dot : array, optional
        Known time derivatives (same shape conventions as ``X``). Rust backend only
        when supplied via PySINDy fallback.
    feature_names : list of str, optional
        Names for each coordinate (e.g. ``["x", "y"]``).
    spec : SINDySpec, optional
        Model configuration. ``spec.backend`` selects ``rust``, ``pysindy``, or
        ``auto`` (Rust when ``ts2net_rs`` is built).

    Returns
    -------
    SINDyResult
        Fitted model, coefficient matrix, and helpers.
    """
    spec = spec or SINDySpec()
    X_norm, n_vars = _normalize_X(X)

    if feature_names is None:
        state_names = [f"x{i}" for i in range(n_vars)]
    else:
        if len(feature_names) != n_vars:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"n_coords ({n_vars})"
            )
        state_names = list(feature_names)

    backend = _resolve_sindy_backend(spec)
    if x_dot is not None and backend == "rust":
        backend = "pysindy"

    if backend == "rust":
        coef, lib_names = _fit_sindy_rust(X_norm, t, state_names=state_names, spec=spec)
        model = None
    else:
        model, coef, lib_names = _fit_sindy_pysindy(
            X_norm, t, x_dot=x_dot, state_names=state_names, spec=spec
        )

    if isinstance(t, (float, int)):
        t_store: NDArray[np.float64] | float | list[NDArray[np.float64]] | None = float(t)
    elif isinstance(t, list):
        t_store = [np.asarray(a, dtype=np.float64) for a in t]
    else:
        t_store = np.asarray(t, dtype=np.float64)

    return SINDyResult(
        model=model,
        coefficients=coef,
        feature_names=lib_names,
        state_names=state_names,
        t=t_store,
        spec=spec,
        backend_used=backend,
    )
