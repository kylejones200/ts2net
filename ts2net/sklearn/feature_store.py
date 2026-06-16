"""
Export network features with stable names and metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class FeatureMetadata:
    """Provenance metadata for exported feature columns."""

    method: str
    builder_params: dict[str, Any] = field(default_factory=dict)
    version: str = "0.8.0"
    source: str = "ts2net"


def features_to_dataframe(
    X_features: NDArray[np.float64],
    feature_names: Sequence[str],
    index: Sequence[Any] | None = None,
    metadata: FeatureMetadata | None = None,
) -> pd.DataFrame:
    """
    Export a feature matrix with stable column names and attrs metadata.

    Parameters
    ----------
    X_features : array (n_samples, n_features)
        Feature matrix from a ts2net sklearn transformer.
    feature_names : sequence of str
        Column names (e.g. from ``get_feature_names_out()``).
    index : sequence, optional
        Row index (series ids, timestamps, etc.).
    metadata : FeatureMetadata, optional
        Stored in ``df.attrs['ts2net']``.

    Returns
    -------
    pandas.DataFrame
        Feature table ready for ML pipelines or Parquet export.

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net.sklearn import NetworkFeatureExtractor, features_to_dataframe
    >>> ext = NetworkFeatureExtractor(method="hvg")
    >>> X = np.random.randn(5, 80)
    >>> feats = ext.fit_transform(X)
    >>> df = features_to_dataframe(feats, ext.get_feature_names_out())
    >>> df.shape[1] == ext.n_features_out_
    True
    """
    df = pd.DataFrame(X_features, columns=list(feature_names), index=index)
    if metadata is not None:
        df.attrs["ts2net"] = {
            "method": metadata.method,
            "builder_params": metadata.builder_params,
            "version": metadata.version,
            "source": metadata.source,
            "n_features": len(feature_names),
        }
    return df
