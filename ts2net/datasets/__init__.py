"""Reference datasets for validation and benchmarks."""

from ts2net.datasets.registry import DatasetSpec, load_dataset, list_datasets
from ts2net.datasets.ucr import (
    list_ucr_datasets,
    load_ucr,
    load_ucr_baselines,
    run_ucr_benchmark,
    validate_ucr_benchmark,
)

__all__ = [
    "DatasetSpec",
    "load_dataset",
    "list_datasets",
    "list_ucr_datasets",
    "load_ucr",
    "load_ucr_baselines",
    "run_ucr_benchmark",
    "validate_ucr_benchmark",
]
