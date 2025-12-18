"""
Distributed computation utilities for large-scale time series analysis.

This module provides tools for parallel and distributed computation of
time series distances and network construction.
"""

import os
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class DistJobConfig:
    """Configuration for distributed distance computation jobs."""

    out_dir: str
    chunk_size: int = 64
    metric: str = "dtw"
    ccf_max_lag: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistJobConfig":
        """Create config from dictionary."""
        return cls(**data)


def _load_panel_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load time series panel from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (data_array, series_names)
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, df.index.tolist()


def ts_dist_part_file(csv_path: str, cfg: DistJobConfig) -> str:
    """
    Compute pairwise distances in shards and write triplets to disk.

    Args:
        csv_path: Path to input CSV file
        cfg: Configuration object

    Returns:
        Path to the manifest file
    """
    # Create output directory
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, names = _load_panel_csv(csv_path)
    n_series = len(X)

    # Calculate chunks
    chunks = [
        (i, min(i + cfg.chunk_size, n_series))
        for i in range(0, n_series, cfg.chunk_size)
    ]
    n_chunks = len(chunks)

    # Save manifest
    manifest = {
        "csv_path": str(Path(csv_path).absolute()),
        "n_series": n_series,
        "chunk_size": cfg.chunk_size,
        "metric": cfg.metric,
        "chunks": [{"start": s, "end": e, "done": False} for s, e in chunks],
        "output_files": [],
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Process chunks
    for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Processing chunks")):
        output_file = out_dir / f"dist_chunk_{chunk_idx:04d}.csv"

        # Compute distances for this chunk
        with open(output_file, "w") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["i", "j", "distance"])

            for i in range(start, end):
                for j in range(i + 1, n_series):
                    # Compute distance based on metric
                    if cfg.metric == "dtw":
                        from ..distances import tsdist_dtw

                        d = tsdist_dtw(X[i : i + 1], X[j : j + 1])[0, 0]
                    elif cfg.metric == "cor":
                        from scipy.stats import pearsonr

                        d = 1 - abs(pearsonr(X[i], X[j])[0])
                    elif cfg.metric == "ccf":
                        from scipy.signal import correlate

                        ccf = correlate(
                            X[i] - X[i].mean(), X[j] - X[j].mean(), mode="full"
                        )
                        max_r = np.max(np.abs(ccf)) / (
                            np.std(X[i]) * np.std(X[j]) * len(X[i])
                        )
                        d = 1 - max_r
                    else:
                        raise ValueError(f"Unsupported metric: {cfg.metric}")

                    writer.writerow([i, j, d])

        # Update manifest
        manifest["chunks"][chunk_idx]["done"] = True
        manifest["output_files"].append(str(output_file))

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return str(manifest_path)


def ts_dist_merge_parts(manifest_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Merge distance chunks into a full distance matrix.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Tuple of (distance_matrix, series_names)
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Initialize distance matrix
    n = manifest["n_series"]
    D = np.zeros((n, n))

    # Load all chunks
    for chunk_file in manifest["output_files"]:
        with open(chunk_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                i, j, d = int(row[0]), int(row[1]), float(row[2])
                D[i, j] = D[j, i] = d

    # Load series names
    df = pd.read_csv(manifest["csv_path"], index_col=0)
    series_names = df.index.tolist()

    return D, series_names
