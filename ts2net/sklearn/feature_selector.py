"""
Feature selection helpers for network-derived features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.validation import check_is_fitted

_SCORE_FUNCS = {
    "f_classif": f_classif,
    "mutual_info": mutual_info_classif,
}


class NetworkFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k network features using univariate scoring.

    Intended for use after :class:`NetworkFeatureExtractor` or
    :class:`RollingNetworkFeatureExtractor` in a sklearn pipeline.

    Parameters
    ----------
    k : int, default 10
        Number of features to retain.
    score_func : {"f_classif", "mutual_info"}, default "mutual_info"
        Univariate scoring function.
    feature_names : list of str, optional
        Names of input features (for ``get_feature_names_out``).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import Pipeline
    >>> from ts2net.sklearn import NetworkFeatureExtractor, NetworkFeatureSelector
    >>> X = np.random.randn(40, 100)
    >>> y = np.array([0] * 20 + [1] * 20)
    >>> ext = NetworkFeatureExtractor(method="hvg")
    >>> Xf = ext.fit_transform(X)
    >>> names = list(ext.get_feature_names_out())
    >>> sel = NetworkFeatureSelector(k=3, feature_names=names)
    >>> sel.fit(Xf, y).transform(Xf).shape[1] == 3
    True
    """

    def __init__(
        self,
        k: int = 10,
        score_func: Literal["f_classif", "mutual_info"] = "mutual_info",
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.k = k
        self.score_func = score_func
        self.feature_names = feature_names

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
    ) -> NetworkFeatureSelector:
        if y is None:
            raise ValueError("NetworkFeatureSelector requires labels y in fit()")
        scorer = _SCORE_FUNCS.get(self.score_func)
        if scorer is None:
            raise ValueError(
                f"Unknown score_func {self.score_func!r}. "
                f"Choose from {sorted(_SCORE_FUNCS)}"
            )
        k = min(self.k, X.shape[1])
        self.selector_ = SelectKBest(score_func=scorer, k=k)
        self.selector_.fit(X, y)
        self.support_ = self.selector_.get_support()
        self.scores_ = self.selector_.scores_
        if self.feature_names is not None:
            names = list(self.feature_names)
            self.selected_features_ = [
                names[i] for i, keep in enumerate(self.support_) if keep
            ]
        else:
            self.selected_features_ = [
                f"feature_{i}" for i, keep in enumerate(self.support_) if keep
            ]
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        check_is_fitted(self, "selector_")
        return self.selector_.transform(X)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
