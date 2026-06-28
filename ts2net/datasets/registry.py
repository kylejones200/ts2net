"""
Curated reference datasets for validation and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPAIN_CSV = _REPO_ROOT / "experiments/spain-multi-meter/spain_meter_network_results.csv"


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a registered reference dataset."""

    name: str
    task: str
    loader: Callable[..., dict[str, Any]]
    citation: str = ""
    description: str = ""
    optional: bool = False


def _synthetic_causal(
    n: int = 2000,
    seed: int = 42,
    r: float = 3.9,
    coupling: float = 0.1,
    **_: Any,
) -> dict[str, Any]:
    """Coupled logistic maps for transfer-entropy validation."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    x[0], y[0] = rng.random(2)

    for t in range(1, n):
        x[t] = (1.0 - coupling) * (r * x[t - 1] * (1.0 - x[t - 1])) + coupling * y[t - 1]
        y[t] = (1.0 - coupling) * (r * y[t - 1] * (1.0 - y[t - 1])) + coupling * x[t - 1]

    X = np.column_stack([x, y])
    return {
        "X": X,
        "y": None,
        "metadata": {
            "name": "synthetic_causal",
            "task": "causality",
            "citation": "Coupled logistic maps (synthetic benchmark)",
            "seed": seed,
            "n": n,
            "r": r,
            "coupling": coupling,
            "n_series": 2,
        },
    }


def _synthetic_classification(
    n_per_class: int = 30,
    n_points: int = 200,
    seed: int = 42,
    **_: Any,
) -> dict[str, Any]:
    """Simple smart-meter-like panel for ML benchmark smoke tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64)
    series: list[NDArray[np.float64]] = []
    labels: list[int] = []

    for label in range(2):
        for _ in range(n_per_class):
            if label == 0:
                x = (
                    0.3
                    + 0.1 * np.sin(2 * np.pi * t / 24)
                    + 0.02 * rng.standard_normal(n_points)
                )
            else:
                x = (
                    1.2
                    + 0.35 * np.sin(2 * np.pi * t / 24)
                    + 0.25 * rng.standard_normal(n_points)
                )
                spikes = rng.choice(n_points, size=12, replace=False)
                x[spikes] += rng.uniform(2, 4, size=12)
            series.append(x.astype(np.float64))
            labels.append(label)

    X = np.vstack(series)
    y = np.array(labels, dtype=np.int64)
    return {
        "X": X,
        "y": y,
        "metadata": {
            "name": "synthetic_classification",
            "task": "classification",
            "citation": "Synthetic consumption patterns (ts2net ML smoke)",
            "seed": seed,
            "n_series": len(series),
            "n_points": n_points,
            "n_classes": 2,
        },
    }


def _spain_meters_summary(**_: Any) -> dict[str, Any]:
    """Bundled Spain meter network summary statistics (if CSV is present)."""
    if not _SPAIN_CSV.is_file():
        raise FileNotFoundError(
            f"Spain meter summary not found at {_SPAIN_CSV}. "
            "Clone the full repo or run the Spain experiment to generate it."
        )

    import pandas as pd

    df = pd.read_csv(_SPAIN_CSV)
    feature_cols = [
        "hvg_avg_degree",
        "hvg_max_degree",
        "nvg_avg_degree",
        "nvg_max_degree",
        "tn_avg_degree",
    ]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    return {
        "X": X,
        "y": None,
        "metadata": {
            "name": "spain_meters_summary",
            "task": "clustering",
            "citation": "Spain multi-meter experiment (bundled summary CSV)",
            "path": str(_SPAIN_CSV),
            "n_meters": len(df),
            "feature_cols": feature_cols,
        },
    }


def _synthetic_regime(
    n: int = 1000,
    seed: int = 42,
    **_: Any,
) -> dict[str, Any]:
    """Piecewise regime panel for dynamic / regime-detection smoke tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = np.where(
        t < n // 2,
        0.2 * np.sin(2 * np.pi * t / 40) + 0.02 * rng.standard_normal(n),
        1.5 * np.sin(2 * np.pi * t / 12) + 0.15 * rng.standard_normal(n),
    ).astype(np.float64)
    labels = np.where(t < n // 2, 0, 1).astype(np.int64)
    return {
        "X": x.reshape(1, -1),
        "y": labels,
        "metadata": {
            "name": "synthetic_regime",
            "task": "regime_detection",
            "citation": "Synthetic piecewise regime shift (ts2net validation)",
            "seed": seed,
            "n_points": n,
            "n_regimes": 2,
        },
    }


def _synthetic_anomaly(
    n: int = 500,
    n_anomalies: int = 8,
    seed: int = 42,
    **_: Any,
) -> dict[str, Any]:
    """Normal baseline series with injected point anomalies."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = 0.5 + 0.1 * np.sin(2 * np.pi * t / 24) + 0.01 * rng.standard_normal(n)
    idx = rng.choice(n, size=n_anomalies, replace=False)
    x[idx] += rng.uniform(3.0, 5.0, size=n_anomalies)
    y = np.zeros(n, dtype=np.int64)
    y[idx] = 1
    return {
        "X": x.reshape(1, -1),
        "y": y,
        "metadata": {
            "name": "synthetic_anomaly",
            "task": "anomaly_detection",
            "citation": "Synthetic point anomalies on periodic baseline",
            "seed": seed,
            "n_points": n,
            "n_anomalies": n_anomalies,
        },
    }


REGISTRY: dict[str, DatasetSpec] = {
    "synthetic_causal": DatasetSpec(
        name="synthetic_causal",
        task="causality",
        citation="Coupled logistic maps (synthetic)",
        description="Coupled logistic maps for transfer-entropy validation.",
        loader=_synthetic_causal,
    ),
    "synthetic_classification": DatasetSpec(
        name="synthetic_classification",
        task="classification",
        citation="Synthetic smart-meter patterns",
        description="Two-class panel for ML benchmark smoke tests.",
        loader=_synthetic_classification,
    ),
    "spain_meters_summary": DatasetSpec(
        name="spain_meters_summary",
        task="clustering",
        citation="Spain multi-meter network experiment",
        description="Per-meter HVG/NVG/TN summary features from bundled CSV.",
        loader=_spain_meters_summary,
        optional=True,
    ),
    "synthetic_regime": DatasetSpec(
        name="synthetic_regime",
        task="regime_detection",
        citation="Synthetic piecewise regime shift",
        description="Single series with two amplitude/frequency regimes.",
        loader=_synthetic_regime,
    ),
    "synthetic_anomaly": DatasetSpec(
        name="synthetic_anomaly",
        task="anomaly_detection",
        citation="Synthetic point anomalies",
        description="Periodic baseline with injected spike anomalies.",
        loader=_synthetic_anomaly,
    ),
}


def list_datasets(*, include_optional: bool = True) -> list[str]:
    """Return registered dataset names."""
    return [
        name
        for name, spec in REGISTRY.items()
        if include_optional or not spec.optional
    ]


def load_dataset(name: str, **kwargs: Any) -> dict[str, Any]:
    """
    Load a registered reference dataset.

    Returns
    -------
    dict
        ``{"X": ndarray, "y": optional, "metadata": {...}}``
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}")

    spec = REGISTRY[name]
    result = spec.loader(**kwargs)
    meta = dict(result.get("metadata", {}))
    meta.setdefault("name", spec.name)
    meta.setdefault("task", spec.task)
    meta.setdefault("citation", spec.citation)
    result["metadata"] = meta
    return result
