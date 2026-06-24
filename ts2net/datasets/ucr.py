"""
UCR/UEA classification benchmark harness (Horizon 9 / v0.9).

Uses aeon/sktime loaders when available; falls back to the bundled
synthetic classification panel for CI smoke tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ts2net.datasets.registry import load_dataset

_UCR_NAMES = ("GunPoint", "ItalyPowerDemand", "Coffee")


def list_ucr_datasets() -> list[str]:
    """Return supported UCR dataset names."""
    return list(_UCR_NAMES)


def load_ucr(
    name: str,
    *,
    return_metadata: bool = False,
) -> tuple[NDArray[np.float64], NDArray[Any]] | tuple[NDArray[np.float64], NDArray[Any], dict[str, Any]]:
    """
    Load a UCR univariate classification dataset.

    Tries ``aeon.datasets.load_classification`` then ``sktime``; on failure
    falls back to ``synthetic_classification`` for smoke testing.
    """
    name = name.strip()
    X: NDArray[np.float64]
    y: NDArray[Any]
    meta: dict[str, Any] = {"name": name, "source": "unknown"}

    try:
        from aeon.datasets import load_classification

        X, y = load_classification(name, split="train", return_type="numpy2d")
        meta["source"] = "aeon"
    except Exception:
        try:
            from sktime.datasets import load_UCR_UEA_dataset

            X, y = load_UCR_UEA_dataset(name, return_X_y=True)
            X = np.asarray(X.squeeze(), dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            y = np.asarray(y)
            meta["source"] = "sktime"
        except Exception:
            data = load_dataset("synthetic_classification", n_per_class=25, n_points=128, seed=0)
            X = data["X"]
            y = data["y"]
            meta["source"] = "synthetic_fallback"
            meta["fallback_reason"] = f"UCR dataset {name!r} not available locally"

    if return_metadata:
        meta.setdefault("n_series", int(X.shape[0]))
        meta.setdefault("n_timesteps", int(X.shape[1]))
        meta.setdefault("n_classes", int(len(np.unique(y))))
        return X, y, meta
    return X, y


def run_ucr_benchmark(
    dataset: str = "GunPoint",
    *,
    cv: int = 5,
    include_optional_baselines: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Cross-validate network vs baseline features on a UCR-style panel.

    Writes JSON results when ``output_path`` is set.
    """
    from ts2net.sklearn import NetworkFeatureExtractor, compare_feature_sets
    from ts2net.sklearn.benchmarks import statistical_baseline_features

    X, y, meta = load_ucr(dataset, return_metadata=True)
    feature_sets: dict[str, NDArray[np.float64]] = {
        "network_hvg": NetworkFeatureExtractor(method="hvg").fit_transform(X),
    }
    stat, _ = statistical_baseline_features(X)
    feature_sets["statistical"] = stat

    if include_optional_baselines:
        from ts2net.sklearn.benchmarks import catch22_baseline_features, sktime_baseline_features

        try:
            c22, _ = catch22_baseline_features(X)
            feature_sets["catch22"] = c22
        except ImportError:
            pass
        try:
            sk, _ = sktime_baseline_features(X)
            feature_sets["sktime"] = sk
        except ImportError:
            pass

    scores = compare_feature_sets(X, y, feature_sets, cv=cv)
    payload = {
        "dataset": dataset,
        "metadata": meta,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cv": cv,
        "scores": scores,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
