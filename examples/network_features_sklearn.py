"""
Classify meter consumption patterns using network features in a sklearn pipeline.

Demonstrates NetworkFeatureExtractor on:
1. Synthetic smart-meter-like time series (always runs, no network required)
2. Spain experiment results bundled in the repo (clustering validation)

Run:
    python examples/network_features_sklearn.py

Optional FRED recession classification (requires internet + examples extra):
    python examples/network_features_sklearn.py --fred
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ts2net.sklearn import NetworkFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPAIN_CSV = os.path.join(
    REPO_ROOT,
    "experiments",
    "spain-multi-meter",
    "spain_meter_network_results.csv",
)


def generate_meter_series(n_per_class: int = 30, n_points: int = 500, seed: int = 42):
    """Synthetic residential vs commercial consumption patterns."""
    rng = np.random.default_rng(seed)
    series = []
    labels = []

    for label, name in enumerate(["residential", "commercial"]):
        for _ in range(n_per_class):
            t = np.arange(n_points)
            if label == 0:
                # Low, regular consumption with daily seasonality
                x = (
                    0.3
                    + 0.1 * np.sin(2 * np.pi * t / 24)
                    + 0.02 * rng.standard_normal(n_points)
                )
            else:
                # Higher, bursty consumption with irregular spikes
                x = (
                    1.5
                    + 0.4 * np.sin(2 * np.pi * t / 24)
                    + 0.3 * rng.standard_normal(n_points)
                )
                spikes = rng.choice(n_points, size=20, replace=False)
                x[spikes] += rng.uniform(2, 5, size=20)
            series.append(x)
            labels.append(label)

    return np.vstack(series), np.array(labels)


def run_synthetic_classification():
    """Train a classifier on network features from synthetic meter data."""
    logger.info("=" * 60)
    logger.info("Synthetic meter classification (NetworkFeatureExtractor + sklearn)")
    logger.info("=" * 60)

    X, y = generate_meter_series()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("net", NetworkFeatureExtractor(method="hvg", output="stats")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(pipe, X, y, cv=5).mean()

    feature_names = pipe.named_steps["net"].get_feature_names_out()
    logger.info("Features: %s", ", ".join(feature_names))
    logger.info("Hold-out accuracy: %.3f", acc)
    logger.info("5-fold CV accuracy: %.3f", cv)
    return acc


def run_spain_clustering():
    """Validate network features on published Spain experiment results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Spain multi-meter case study (bundled experiment results)")
    logger.info("=" * 60)

    if not os.path.exists(SPAIN_CSV):
        logger.warning("Spain results CSV not found at %s — skipping.", SPAIN_CSV)
        return

    df = pd.read_csv(SPAIN_CSV)
    feature_cols = [
        "hvg_avg_degree",
        "nvg_avg_degree",
        "tn_avg_degree",
        "std_consumption",
        "mean_consumption",
    ]
    X = df[feature_cols].values

    # K=2: low vs high NVG complexity (matches README finding)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = km.fit_predict(X)
    sil = silhouette_score(X, clusters)
    df["cluster"] = clusters

    logger.info("Meters analyzed: %d", len(df))
    logger.info(
        "HVG avg degree: %.3f (theory ≈ 4.0)",
        df["hvg_avg_degree"].mean(),
    )
    logger.info(
        "NVG avg degree range: %.2f – %.2f",
        df["nvg_avg_degree"].min(),
        df["nvg_avg_degree"].max(),
    )
    logger.info("KMeans silhouette (k=2 on network features): %.3f", sil)
    logger.info("Cluster means:\n%s", df.groupby("cluster")[feature_cols].mean().round(2).to_string())


def run_fred_classification():
    """Optional: classify recession periods from FRED unemployment series."""
    try:
        import pandas_datareader.data as web
    except ImportError:
        logger.error(
            "pandas-datareader required. Install with: pip install ts2net[examples]"
        )
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("FRED recession classification (UNRATE network features)")
    logger.info("=" * 60)

    unrate = web.DataReader("UNRATE", "fred", start="1990-01-01")
    usrec = web.DataReader("USREC", "fred", start="1990-01-01")
    df = unrate.join(usrec, how="inner").dropna()
    df.columns = ["unrate", "recession"]

    window = 36  # 3 years of monthly data
    series = []
    labels = []
    values = df["unrate"].values
    rec = df["recession"].values.astype(int)

    for i in range(len(values) - window):
        series.append(values[i : i + window])
        labels.append(int(rec[i + window] > 0))

    X = np.vstack(series)
    y = np.array(labels)

    pipe = Pipeline(
        [
            ("net", NetworkFeatureExtractor(method="transition", output="stats")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    logger.info("Windows: %d (recession rate %.1f%%)", len(y), 100 * y.mean())
    logger.info("5-fold CV accuracy: %.3f ± %.3f", scores.mean(), scores.std())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fred",
        action="store_true",
        help="Also run optional FRED recession classification (needs internet)",
    )
    args = parser.parse_args()

    run_synthetic_classification()
    run_spain_clustering()
    if args.fred:
        run_fred_classification()


if __name__ == "__main__":
    main()
