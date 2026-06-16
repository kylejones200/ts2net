"""
Baseline feature extractors and benchmark comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def statistical_baseline_features(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Classical time-series statistics as a baseline feature set.

    Per series: mean, std, skew, kurtosis, lag-1 autocorrelation,
    trend slope, and zero-crossing rate.

    Parameters
    ----------
    X : array (n_series, n_timesteps)
        Panel of univariate time series.

    Returns
    -------
    features : array (n_series, n_features)
    names : list of str
        Stable feature names.
    """
    X = np.asarray(X, dtype=np.float64)
    n_series = X.shape[0]
    names = [
        "stat_mean",
        "stat_std",
        "stat_skew",
        "stat_kurtosis",
        "stat_autocorr_lag1",
        "stat_trend_slope",
        "stat_zero_crossing_rate",
    ]
    rows = []
    for i in range(n_series):
        x = X[i]
        x = x[np.isfinite(x)]
        if len(x) < 3:
            rows.append([0.0] * len(names))
            continue
        mean = float(np.mean(x))
        std = float(np.std(x))
        skew = float(sp_stats.skew(x)) if std > 0 else 0.0
        kurt = float(sp_stats.kurtosis(x)) if std > 0 else 0.0
        if len(x) > 1 and std > 0:
            ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        else:
            ac1 = 0.0
        t = np.arange(len(x), dtype=np.float64)
        slope = float(np.polyfit(t, x, 1)[0]) if std > 0 else 0.0
        zcr = float(np.mean(np.diff(np.signbit(x - mean))))
        rows.append([mean, std, skew, kurt, ac1, slope, zcr])
    return np.asarray(rows, dtype=np.float64), names


def tsfresh_baseline_features(
    X: NDArray[np.float64],
    column_id: str = "id",
    column_sort: str = "time",
    column_value: str = "value",
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Extract tsfresh features (optional dependency).

  Requires ``pip install ts2net[tsfresh]`` or ``pip install tsfresh``.

    Parameters
    ----------
    X : array (n_series, n_timesteps)
        Panel of time series.

    Returns
    -------
    features, feature_names
    """
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError as exc:
        raise ImportError(
            "tsfresh required for tsfresh_baseline_features. "
            "Install with: pip install ts2net[tsfresh]"
        ) from exc

    rows = []
    for i, series in enumerate(X):
        for t, val in enumerate(series):
            rows.append({column_id: i, column_sort: t, column_value: float(val)})
    import pandas as pd

    df = pd.DataFrame(rows)
    extracted = extract_features(
        df,
        column_id=column_id,
        column_sort=column_sort,
        column_value=column_value,
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=True,
    )
    extracted = extracted.sort_index()
    extracted = extracted.fillna(0.0)
    return extracted.values.astype(np.float64), list(extracted.columns)


def compare_feature_sets(
    X: NDArray[np.float64],
    y: NDArray[Any],
    feature_sets: dict[str, NDArray[np.float64]],
    estimator: Any | None = None,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict[str, dict[str, float]]:
    """
    Cross-validate classifiers on multiple feature representations.

    Parameters
    ----------
    X : array (n_series, n_timesteps)
        Raw time series (unused if all sets are precomputed).
    y : array (n_series,)
        Class labels.
    feature_sets : dict
        Mapping name → feature matrix ``(n_series, n_features)``.
    estimator : sklearn estimator, optional
        Classifier (default: scaled logistic regression).
    cv : int, default 5
        Cross-validation folds.
    scoring : str, default "accuracy"
        sklearn scoring metric.

    Returns
    -------
    dict
        Per feature set: ``mean_score``, ``std_score``, ``n_features``.

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.sklearn import NetworkFeatureExtractor, compare_feature_sets
    >>> from ts2net.sklearn.benchmarks import statistical_baseline_features
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((40, 120))
    >>> y = np.array([0] * 20 + [1] * 20)
    >>> net = NetworkFeatureExtractor(method="hvg").fit_transform(X)
    >>> base, _ = statistical_baseline_features(X)
    >>> results = compare_feature_sets(
    ...     X, y, {"network": net, "statistical": base}
    ... )
    >>> "network" in results
    True
    """
    if estimator is None:
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

    results: dict[str, dict[str, float]] = {}
    for name, Xf in feature_sets.items():
        Xf = np.asarray(Xf, dtype=np.float64)
        if Xf.shape[0] != len(y):
            raise ValueError(
                f"feature set {name!r} has {Xf.shape[0]} rows but y has {len(y)}"
            )
        scores = cross_val_score(
            clone(estimator), Xf, y, cv=cv, scoring=scoring
        )
        results[name] = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "n_features": int(Xf.shape[1]),
        }
    return results
