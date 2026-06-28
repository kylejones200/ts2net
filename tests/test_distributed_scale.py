"""Tests for Dask/Ray distributed scale helpers (Horizon 4 / v0.6)."""

from __future__ import annotations

import numpy as np
import pytest

from ts2net import build_windows
from ts2net.multivariate.distances import ts_dist
from ts2net.scale.distributed import (
    build_windows_distributed,
    dask_available,
    parallel_map,
    ts_dist_distributed,
)


class TestParallelMap:
    def test_joblib_executor(self):
        out = parallel_map(lambda x: x * 2, [(1,), (2,), (3,)], executor="joblib")
        assert out == [2, 4, 6]


class TestDistributedTsDist:
    def test_dask_matches_serial(self):
        pytest.importorskip("dask")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((24, 35))
        D_serial = ts_dist(X, method="correlation", n_jobs=1)
        D_dask = ts_dist(
            X,
            method="correlation",
            executor="dask",
            row_chunk_size=8,
        )
        np.testing.assert_allclose(D_serial, D_dask, rtol=1e-10, atol=1e-10)

    def test_ts_dist_distributed_direct(self):
        pytest.importorskip("dask")
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 25))
        D = ts_dist_distributed(X, method="correlation", executor="dask", row_chunk_size=5)
        assert D.shape == (20, 20)


class TestDistributedBuildWindows:
    def test_dask_matches_serial(self):
        pytest.importorskip("dask")
        rng = np.random.default_rng(2)
        x = rng.standard_normal(300)
        serial = build_windows(x, window=30, step=15, method="hvg")
        parallel = build_windows_distributed(
            x, window=30, step=15, method="hvg", executor="dask"
        )
        np.testing.assert_array_equal(serial["n_edges"], parallel["n_edges"])

    @pytest.mark.skipif(not dask_available(), reason="dask not installed")
    def test_build_windows_executor_param(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(250)
        a = build_windows(x, window=25, step=10, method="hvg", executor="dask")
        b = build_windows(x, window=25, step=10, method="hvg", n_jobs=1)
        np.testing.assert_array_equal(a["avg_degree"], b["avg_degree"])
