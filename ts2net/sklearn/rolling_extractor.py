"""
Rolling window network features for sklearn pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ts2net.api_windows import build_windows

_AGG_FUNCS = ("mean", "std", "min", "max")


class RollingNetworkFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract network features from rolling windows, aggregated per series.

    Each input row is one time series. Windows are built along the series,
    graph statistics are computed per window, then aggregated (mean, std, etc.)
    into a fixed-length feature vector suitable for sklearn.

    Parameters
    ----------
    method : str, default "hvg"
        Network method: ``hvg``, ``nvg``, ``recurrence``, ``transition``.
    window : int, default 64
        Window width in time points.
    step : int, default 32
        Step between consecutive windows.
    aggregates : list of str, default ("mean", "std")
        Aggregations applied to each window-level statistic.
    features : list of str, optional
        Window stats to include (default: core graph stats).
    prefix : str, optional
        Feature name prefix.
    **builder_kwargs
        Extra arguments for the network builder.

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.sklearn import RollingNetworkFeatureExtractor
    >>> X = np.random.randn(20, 300)
    >>> ext = RollingNetworkFeatureExtractor(window=50, step=25)
    >>> ext.fit(X).transform(X).shape[0] == 20
    True
    """

    def __init__(
        self,
        method: str = "hvg",
        window: int = 64,
        step: int = 32,
        aggregates: Sequence[str] = ("mean", "std"),
        features: Sequence[str] | None = None,
        prefix: str | None = None,
        **builder_kwargs: Any,
    ) -> None:
        self.method = method
        self.window = window
        self.step = step
        self.aggregates = tuple(aggregates)
        self.features = features
        self.prefix = prefix
        self.builder_kwargs = builder_kwargs

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any] | None = None,
    ) -> RollingNetworkFeatureExtractor:
        X = self._validate_X(X)
        for agg in self.aggregates:
            if agg not in _AGG_FUNCS:
                raise ValueError(f"aggregate must be one of {_AGG_FUNCS}, got {agg!r}")

        probe = self._series_features(X[0])
        self.feature_names_ = list(probe.keys())
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, "feature_names_")
        X = self._validate_X(X)
        return np.vstack([self._series_to_vector(x) for x in X])

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    def _validate_X(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            X = check_array(
                X, dtype="numeric", ensure_2d=True, ensure_all_finite=False
            )
        except TypeError:
            X = check_array(
                X, dtype="numeric", ensure_2d=True, force_all_finite=False
            )
        if X.shape[1] < self.window:
            raise ValueError(
                f"Series length {X.shape[1]} must be >= window={self.window}"
            )
        return X

    def _stat_names(self) -> list[str]:
        if self.features is not None:
            return list(self.features)
        return ["n_nodes", "n_edges", "avg_degree", "std_degree", "density"]

    def _series_features(self, x: NDArray[np.float64]) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        window_stats = build_windows(
            x,
            window=self.window,
            step=self.step,
            method=self.method,
            output="stats",
            **self.builder_kwargs,
        )
        stat_names = self._stat_names()
        prefix = self.prefix or f"{self.method}_roll_"
        out: dict[str, float] = {}

        for stat in stat_names:
            if stat not in window_stats:
                continue
            arr = np.asarray(window_stats[stat], dtype=np.float64)
            for agg in self.aggregates:
                if agg == "mean":
                    val = float(np.mean(arr))
                elif agg == "std":
                    val = float(np.std(arr)) if len(arr) > 1 else 0.0
                elif agg == "min":
                    val = float(np.min(arr))
                else:
                    val = float(np.max(arr))
                out[f"{prefix}{stat}_{agg}"] = val
        return out

    def _series_to_vector(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        feats = self._series_features(x)
        return np.array(
            [feats[name] for name in self.feature_names_], dtype=np.float64
        )
