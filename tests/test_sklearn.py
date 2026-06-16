"""Tests for sklearn integration."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ts2net.sklearn import NetworkFeatureExtractor


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 80))
    y = np.array([0] * 6 + [1] * 6)
    return X, y


class TestNetworkFeatureExtractor:
    def test_fit_transform_shape(self, panel_data):
        X, _ = panel_data
        ext = NetworkFeatureExtractor(method="hvg")
        ext.fit(X)
        out = ext.transform(X)
        assert out.shape == (X.shape[0], ext.n_features_out_)
        assert out.shape[1] == len(ext.get_feature_names_out())

    def test_feature_names_prefixed(self, panel_data):
        X, _ = panel_data
        ext = NetworkFeatureExtractor(method="hvg", prefix="test_")
        ext.fit(X)
        names = ext.get_feature_names_out()
        assert all(name.startswith("test_") for name in names)

    def test_pipeline_integration(self, panel_data):
        X, y = panel_data
        pipe = Pipeline(
            [
                ("net", NetworkFeatureExtractor(method="hvg")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        pipe.fit(X, y)
        assert pipe.score(X, y) > 0.5

    def test_nvg_with_limit(self, panel_data):
        X, _ = panel_data
        ext = NetworkFeatureExtractor(method="nvg", limit=50, output="stats")
        out = ext.fit_transform(X)
        assert out.shape[0] == X.shape[0]
        assert np.all(np.isfinite(out))

    def test_unknown_method_raises(self, panel_data):
        X, _ = panel_data
        ext = NetworkFeatureExtractor(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            ext.fit(X)

    def test_too_short_series_raises(self):
        X = np.random.randn(2, 2)
        ext = NetworkFeatureExtractor()
        with pytest.raises(ValueError, match="at least 3"):
            ext.fit(X)

    def test_custom_features_subset(self, panel_data):
        X, _ = panel_data
        ext = NetworkFeatureExtractor(
            method="hvg", features=["n_edges", "avg_degree"]
        )
        out = ext.fit_transform(X)
        assert out.shape[1] == 2
        names = ext.get_feature_names_out()
        assert "hvg_n_edges" in names
        assert "hvg_avg_degree" in names

    def test_import_from_package(self):
        from ts2net import NetworkFeatureExtractor as NFE
        assert NFE is NetworkFeatureExtractor
