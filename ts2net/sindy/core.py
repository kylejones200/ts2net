"""
PySINDy integration — sparse identification of nonlinear dynamics.

Wraps `pysindy` with ts2net conventions for multivariate time series input
``(n_timepoints, n_coordinates)`` and optional conversion to coupling networks.

See the PySINDy tutorial:
https://pysindy.readthedocs.io/en/latest/examples/tutorial_1/example.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series


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


@dataclass
class SINDyResult:
    """Outcome of :func:`fit_sindy`."""

    model: Any
    coefficients: NDArray[np.float64]
    feature_names: list[str]
    state_names: list[str]
    t: NDArray[np.float64] | float | list[NDArray[np.float64]] | None = None
    spec: SINDySpec = field(default_factory=SINDySpec)

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
        Known time derivatives (same shape conventions as ``X``).
    feature_names : list of str, optional
        Names for each coordinate (e.g. ``["x", "y"]``).
    spec : SINDySpec, optional
        Model configuration.

    Returns
    -------
    SINDyResult
        Fitted model, coefficient matrix, and helpers.

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.sindy import SINDySpec, fit_sindy
    >>> t = np.linspace(0, 1, 100)
    >>> X = np.column_stack([3 * np.exp(-2 * t), 0.5 * np.exp(t)])
    >>> result = fit_sindy(X, t, feature_names=["x", "y"], spec=SINDySpec(polynomial_degree=1))
    >>> result.coefficients[0, 1]  # dx/dt coefficient on x
    -2.0...
    """
    _require_pysindy()
    import pysindy as ps

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
    )
