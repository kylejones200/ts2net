"""
sklearn-compatible transformers for network feature extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ts2net import HVG, NVG, RecurrenceNetwork, TransitionNetwork

_METHOD_BUILDERS = {
    "hvg": HVG,
    "nvg": NVG,
    "recurrence": RecurrenceNetwork,
    "transition": TransitionNetwork,
}

_DEFAULT_STATS = [
    "n_nodes",
    "n_edges",
    "avg_degree",
    "std_degree",
    "min_degree",
    "max_degree",
    "density",
]


class NetworkFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract network summary features from time series for sklearn pipelines.

    Each input row is treated as one univariate time series. The transformer
    builds a network with the chosen method and returns summary statistics
    as a numeric feature vector.

    Parameters
    ----------
    method : {"hvg", "nvg", "recurrence", "transition"}, default "hvg"
        Network construction method.
    output : {"stats", "degrees"}, default "stats"
        Builder output mode. ``"stats"`` is memory-efficient and recommended
        for panel data.
    features : list of str, optional
        Subset of summary statistics to return. Defaults to all available stats.
    prefix : str, optional
        Prefix for feature names (e.g. ``"hvg_"``).
    **builder_kwargs
        Additional keyword arguments passed to the network builder
        (e.g. ``limit=2000`` for NVG, ``rule="knn", k=5`` for recurrence).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from ts2net.sklearn import NetworkFeatureExtractor
    >>> X = np.random.randn(40, 200)
    >>> y = np.array([0] * 20 + [1] * 20)
    >>> pipe = Pipeline([
    ...     ("net", NetworkFeatureExtractor(method="hvg")),
    ...     ("scale", StandardScaler()),
    ...     ("clf", LogisticRegression(max_iter=500)),
    ... ])
    >>> pipe.fit(X, y).score(X, y)  # doctest: +SKIP
    """

    def __init__(
        self,
        method: str = "hvg",
        output: str = "stats",
        features: Sequence[str] | None = None,
        prefix: str | None = None,
        **builder_kwargs: Any,
    ) -> None:
        self.method = method
        self.output = output
        self.features = features
        self.prefix = prefix
        self.builder_kwargs = builder_kwargs

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any] | None = None,
    ) -> NetworkFeatureExtractor:
        """Learn feature names from a representative sample."""
        X = self._validate_X(X)
        if self.method not in _METHOD_BUILDERS:
            raise ValueError(
                f"Unknown method {self.method!r}. "
                f"Choose from {sorted(_METHOD_BUILDERS)}"
            )

        probe = self._extract_series_features(X[0])
        self.feature_names_ = list(probe.keys())
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(self.feature_names_)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract network features for each time series."""
        check_is_fitted(self, "feature_names_")
        X = self._validate_X(X)
        rows = [self._series_to_vector(x) for x in X]
        return np.vstack(rows)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return output feature names for sklearn >= 1.0."""
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    def _validate_X(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            X = check_array(
                X, dtype="numeric", ensure_2d=True, ensure_all_finite=False
            )
        except TypeError:
            # sklearn < 1.6
            X = check_array(
                X, dtype="numeric", ensure_2d=True, force_all_finite=False
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X.shape[1] < 3:
            raise ValueError(
                f"Each time series must have at least 3 points, got {X.shape[1]}"
            )
        return X

    def _create_builder(self):
        builder_cls = _METHOD_BUILDERS[self.method]
        return builder_cls(output=self.output, **self.builder_kwargs)

    def _extract_series_features(self, x: NDArray[np.float64]) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 3:
            raise ValueError("Time series must have at least 3 finite values")

        builder = self._create_builder()
        builder.build(x)

        if self.output == "stats":
            stats = builder.stats()
        elif self.output == "degrees":
            degrees = builder.degree_sequence()
            stats = {
                "n_nodes": float(builder.n_nodes),
                "n_edges": float(builder.n_edges),
                "avg_degree": float(np.mean(degrees)),
                "std_degree": float(np.std(degrees)) if len(degrees) > 1 else 0.0,
                "min_degree": float(np.min(degrees)),
                "max_degree": float(np.max(degrees)),
                "density": float(builder.n_edges)
                / max(builder.n_nodes * (builder.n_nodes - 1) / 2, 1),
            }
        else:
            raise ValueError(
                f"Unsupported output mode {self.output!r}. Use 'stats' or 'degrees'."
            )

        selected = self.features if self.features is not None else _DEFAULT_STATS
        prefix = self.prefix or f"{self.method}_"
        return {f"{prefix}{key}": float(stats[key]) for key in selected if key in stats}

    def _series_to_vector(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        features = self._extract_series_features(x)
        return np.array(
            [features[name] for name in self.feature_names_], dtype=np.float64
        )
