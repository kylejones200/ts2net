"""Tests for the reference dataset registry (Horizon 9)."""

from __future__ import annotations

import numpy as np
import pytest

from ts2net.datasets import list_datasets, load_dataset


class TestDatasetRegistry:
    def test_list_datasets(self):
        names = list_datasets()
        assert "synthetic_causal" in names
        assert "synthetic_classification" in names

    def test_synthetic_causal(self):
        data = load_dataset("synthetic_causal", n=100, seed=1)
        assert data["X"].shape == (100, 2)
        assert data["metadata"]["task"] == "causality"

    def test_synthetic_classification(self):
        data = load_dataset("synthetic_classification", n_per_class=5, n_points=50, seed=1)
        assert data["X"].shape[0] == 10
        assert data["y"] is not None
        assert len(np.unique(data["y"])) == 2

    def test_spain_optional(self):
        from pathlib import Path

        csv = Path("experiments/spain-multi-meter/spain_meter_network_results.csv")
        if not csv.is_file():
            pytest.skip("Spain meter summary CSV not bundled")
        data = load_dataset("spain_meters_summary")
        assert data["X"].ndim == 2
        assert data["metadata"]["n_meters"] > 0
