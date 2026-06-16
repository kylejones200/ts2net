"""Tests for ML integration (horizon 0.7)."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ts2net import HVG, NetworkFeatureExtractor
from ts2net.sklearn import (
    RollingNetworkFeatureExtractor,
    NetworkFeatureSelector,
    features_to_dataframe,
    FeatureMetadata,
    statistical_baseline_features,
    compare_feature_sets,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    n = 40
    t = np.arange(200)
    X = []
    y = []
    for i in range(n):
        if i < n // 2:
            x = 0.2 * np.sin(2 * np.pi * t / 20) + 0.05 * rng.standard_normal(200)
            y.append(0)
        else:
            x = 1.5 * np.sin(2 * np.pi * t / 20) + 0.4 * rng.standard_normal(200)
            spikes = rng.choice(200, size=10, replace=False)
            x[spikes] += rng.uniform(2, 5, size=10)
            y.append(1)
        X.append(x)
    return np.vstack(X), np.array(y)


class TestRollingExtractor:
    def test_rolling_features_shape(self, panel):
        X, _ = panel
        ext = RollingNetworkFeatureExtractor(window=50, step=25, aggregates=("mean",))
        out = ext.fit_transform(X)
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] > 0
        assert all("roll" in n for n in ext.get_feature_names_out())


class TestFeatureSelector:
    def test_selects_k_features(self, panel):
        X, y = panel
        ext = NetworkFeatureExtractor(method="hvg")
        Xf = ext.fit_transform(X)
        sel = NetworkFeatureSelector(
            k=3, feature_names=list(ext.get_feature_names_out())
        )
        out = sel.fit_transform(Xf, y)
        assert out.shape[1] == 3
        assert len(sel.get_feature_names_out()) == 3


class TestFeatureStore:
    def test_dataframe_export(self, panel):
        X, _ = panel
        ext = NetworkFeatureExtractor(method="hvg")
        feats = ext.fit_transform(X)
        df = features_to_dataframe(
            feats,
            ext.get_feature_names_out(),
            metadata=FeatureMetadata(method="hvg"),
        )
        assert df.shape == feats.shape
        assert "ts2net" in df.attrs


class TestBenchmarks:
    def test_statistical_baseline(self, panel):
        X, _ = panel
        feats, names = statistical_baseline_features(X)
        assert feats.shape[0] == X.shape[0]
        assert len(names) == feats.shape[1]

    def test_compare_feature_sets(self, panel):
        X, y = panel
        net = NetworkFeatureExtractor(method="hvg").fit_transform(X)
        base, _ = statistical_baseline_features(X)
        results = compare_feature_sets(
            X, y, {"network": net, "statistical": base}, cv=3
        )
        assert set(results) == {"network", "statistical"}
        for r in results.values():
            assert "mean_score" in r
            assert 0 <= r["mean_score"] <= 1.0


class TestPyGAdapter:
    def test_to_pyg_data(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
        from ts2net.ml import to_pyg_data

        hvg = HVG()
        hvg.build(np.random.randn(60))
        data = to_pyg_data(hvg._graph, y=1)
        assert data.x.shape[0] == hvg.n_nodes
        assert data.edge_index.shape[0] == 2
        assert int(data.y.item()) == 1


class TestDGLAdapter:
    def test_to_dgl_graph(self):
        pytest.importorskip("torch")
        dgl = pytest.importorskip("dgl")
        from ts2net.ml import to_dgl_graph

        hvg = HVG()
        hvg.build(np.random.randn(60))
        g = to_dgl_graph(hvg._graph)
        assert g.num_nodes() == hvg.n_nodes
        assert "feat" in g.ndata


class TestMLPipeline:
    def test_full_pipeline_with_selector(self, panel):
        X, y = panel
        ext = NetworkFeatureExtractor(method="hvg")
        pipe = Pipeline(
            [
                ("net", ext),
                (
                    "sel",
                    NetworkFeatureSelector(k=4),
                ),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        # Selector needs feature names after net fit - fit in two steps
        Xf = ext.fit_transform(X)
        sel = NetworkFeatureSelector(
            k=4, feature_names=list(ext.get_feature_names_out())
        )
        Xs = sel.fit_transform(Xf, y)
        clf = LogisticRegression(max_iter=500)
        clf.fit(StandardScaler().fit_transform(Xs), y)
        assert clf.score(StandardScaler().fit_transform(Xs), y) > 0.5
