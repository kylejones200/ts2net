"""Tests for threshold sensitivity sweeps."""

from __future__ import annotations

import numpy as np

from ts2net.stats.threshold_sensitivity import threshold_sensitivity_sweep


class TestThresholdSensitivity:
    def test_recurrence_monotonic_edges(self):
        x = np.sin(np.linspace(0, 8 * np.pi, 200))
        df = threshold_sensitivity_sweep(
            x,
            method="recurrence",
            thresholds=[0.05, 0.2, 0.5, 1.0],
        )
        assert list(df["n_edges"]) == sorted(df["n_edges"].tolist())

    def test_nvg_limit_sweep(self):
        x = np.random.default_rng(0).standard_normal(300)
        df = threshold_sensitivity_sweep(
            x,
            method="nvg",
            thresholds=[50, 100, 200],
        )
        assert len(df) == 3
        assert "avg_degree" in df.columns
