"""
scikit-learn integration for ts2net.

Provides transformers that extract network-based features from time series
for use in sklearn pipelines.
"""

from .benchmarks import (
    catch22_baseline_features,
    compare_feature_sets,
    sktime_baseline_features,
    statistical_baseline_features,
    tsfresh_baseline_features,
)
from .feature_extractor import NetworkFeatureExtractor
from .feature_selector import NetworkFeatureSelector
from .feature_store import FeatureMetadata, features_to_dataframe
from .rolling_extractor import RollingNetworkFeatureExtractor

__all__ = [
    "NetworkFeatureExtractor",
    "RollingNetworkFeatureExtractor",
    "NetworkFeatureSelector",
    "FeatureMetadata",
    "features_to_dataframe",
    "statistical_baseline_features",
    "tsfresh_baseline_features",
    "catch22_baseline_features",
    "sktime_baseline_features",
    "compare_feature_sets",
]
