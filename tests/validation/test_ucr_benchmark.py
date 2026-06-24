"""Tests for UCR benchmark harness."""

from __future__ import annotations

import pytest

from ts2net.datasets import list_ucr_datasets, load_ucr, run_ucr_benchmark


class TestUCRHarness:
    def test_list_ucr(self):
        names = list_ucr_datasets()
        assert "GunPoint" in names

    def test_load_ucr_fallback(self):
        X, y, meta = load_ucr("GunPoint", return_metadata=True)
        assert X.ndim == 2
        assert len(y) == X.shape[0]
        assert meta["source"] in {"aeon", "sktime", "synthetic_fallback"}

    def test_run_ucr_benchmark_smoke(self, tmp_path):
        out = tmp_path / "ucr.json"
        payload = run_ucr_benchmark("GunPoint", cv=3, output_path=out)
        assert "network_hvg" in payload["scores"]
        assert out.is_file()
