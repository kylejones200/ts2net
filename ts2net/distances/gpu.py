"""
GPU-accelerated distance kernels (Horizon 4 / v0.6).

Correlation distance matrices use PyTorch or CuPy when available.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.gpu_backend import resolve_gpu_backend, resolve_torch_device


def cdist_correlation(
    X: NDArray[np.float64],
    *,
    device: str = "auto",
    backend: str = "auto",
) -> NDArray[np.float64]:
    """
    Pairwise correlation distance matrix ``d = 1 - |ρ|``.

    Uses PyTorch or CuPy when requested/available; falls back to NumPy.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    resolved = resolve_gpu_backend(backend, device=device)
    if resolved == "torch":
        return _cdist_correlation_torch(X, device=device)
    if resolved == "cupy":
        return _cdist_correlation_cupy(X)
    return _cdist_correlation_numpy(X)


def _cdist_correlation_numpy(X: NDArray[np.float64]) -> NDArray[np.float64]:
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            rho = np.corrcoef(X[i], X[j])[0, 1]
            d = 1.0 - abs(float(rho))
            D[i, j] = D[j, i] = d
    return D


def _cdist_correlation_torch(
    X: NDArray[np.float64],
    *,
    device: str = "auto",
) -> NDArray[np.float64]:
    import torch

    dev = resolve_torch_device(device)
    t = torch.as_tensor(X, dtype=torch.float64, device=dev)
    t = t - t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True).clamp(min=1e-12)
    t_norm = t / std
    corr = (t_norm @ t_norm.T) / t.shape[1]
    D = 1.0 - corr.abs()
    D.fill_diagonal_(0.0)
    return D.cpu().numpy()


def _cdist_correlation_cupy(X: NDArray[np.float64]) -> NDArray[np.float64]:
    import cupy as cp

    t = cp.asarray(X, dtype=cp.float64)
    t = t - t.mean(axis=1, keepdims=True)
    std = cp.maximum(t.std(axis=1, keepdims=True), 1e-12)
    t_norm = t / std
    corr = (t_norm @ t_norm.T) / t.shape[1]
    D = 1.0 - cp.abs(corr)
    cp.fill_diagonal(D, 0.0)
    return cp.asnumpy(D)
