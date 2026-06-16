"""
ML integration example (horizon 0.7).

Demonstrates rolling network features, feature selection, baseline
comparison, and optional PyG/DGL export.

Run:
    python examples/ml_integration_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ts2net import HVG, NetworkFeatureExtractor
from ts2net.sklearn import (
    RollingNetworkFeatureExtractor,
    NetworkFeatureSelector,
    features_to_dataframe,
    FeatureMetadata,
    statistical_baseline_features,
    compare_feature_sets,
)


def generate_panel(n_per_class: int = 30, n_points: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    series, labels = [], []
    for label in range(2):
        for _ in range(n_per_class):
            if label == 0:
                x = 0.3 + 0.1 * np.sin(2 * np.pi * t / 24) + 0.02 * rng.standard_normal(n_points)
            else:
                x = 1.2 + 0.35 * np.sin(2 * np.pi * t / 24) + 0.25 * rng.standard_normal(n_points)
                spikes = rng.choice(n_points, size=12, replace=False)
                x[spikes] += rng.uniform(2, 4, size=12)
            series.append(x)
            labels.append(label)
    return np.vstack(series), np.array(labels)


def main():
    X, y = generate_panel()
    print("=" * 60)
    print("Feature benchmark: network vs statistical baseline")
    print("=" * 60)

    net_feats = NetworkFeatureExtractor(method="hvg").fit_transform(X)
    roll_feats = RollingNetworkFeatureExtractor(
        method="hvg", window=50, step=25, aggregates=("mean", "std")
    ).fit_transform(X)
    stat_feats, stat_names = statistical_baseline_features(X)

    results = compare_feature_sets(
        X,
        y,
        {
            "network": net_feats,
            "rolling_network": roll_feats,
            "statistical": stat_feats,
        },
        cv=5,
    )
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean_score"]):
        print(
            f"  {name:18s}  accuracy={r['mean_score']:.3f} ± {r['std_score']:.3f}  "
            f"({r['n_features']} features)"
        )

    print()
    print("=" * 60)
    print("Pipeline with feature selection")
    print("=" * 60)
    ext = NetworkFeatureExtractor(method="hvg")
    Xf = ext.fit_transform(X)
    sel = NetworkFeatureSelector(
        k=4, feature_names=list(ext.get_feature_names_out())
    )
    Xs = sel.fit_transform(Xf, y)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    scores = cross_val_score(pipe, Xs, y, cv=5)
    print(f"Selected features: {list(sel.get_feature_names_out())}")
    print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    print()
    print("=" * 60)
    print("Feature store export")
    print("=" * 60)
    df = features_to_dataframe(
        net_feats,
        ext.get_feature_names_out(),
        index=[f"series_{i}" for i in range(len(X))],
        metadata=FeatureMetadata(method="hvg", builder_params={"output": "stats"}),
    )
    print(df.head(3).to_string())
    print(f"Metadata: {df.attrs.get('ts2net', {})}")

    print()
    print("=" * 60)
    print("PyG / DGL export (optional)")
    print("=" * 60)
    try:
        from ts2net.ml import to_pyg_data

        hvg = HVG().build(X[0])
        data = to_pyg_data(hvg._graph, y=int(y[0]))
        print(f"PyG Data: {data.num_nodes} nodes, {data.num_edges} edges")
    except ImportError:
        print("PyG not installed — pip install ts2net[pyg]")

    try:
        from ts2net.ml import to_dgl_graph

        hvg = HVG().build(X[1])
        g = to_dgl_graph(hvg._graph)
        print(f"DGL graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    except ImportError:
        print("DGL not installed — pip install ts2net[dgl]")


if __name__ == "__main__":
    main()
