"""Tests for GPU scale backends (Horizon 4 / v0.6)."""

from __future__ import annotations

import numpy as np
import pytest

from ts2net.core.gpu_backend import resolve_gpu_backend, torch_available
from ts2net.distances.gpu import cdist_correlation
from ts2net.multivariate.distances import ts_dist


class TestGpuBackend:
    def test_resolve_cpu(self):
        assert resolve_gpu_backend("cpu") == "cpu"

    @pytest.mark.skipif(not torch_available(), reason="torch not installed")
    def test_cdist_correlation_torch_matches_numpy(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 40))
        D_np = cdist_correlation(X, backend="cpu")
        D_torch = cdist_correlation(X, backend="torch", device="cpu")
        np.testing.assert_allclose(D_np, D_torch, rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not torch_available(), reason="torch not installed")
    def test_ts_dist_gpu_correlation(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((8, 30))
        D = ts_dist(X, method="correlation", device="gpu", gpu_backend="torch")
        assert D.shape == (8, 8)
        assert np.allclose(np.diag(D), 0.0)
