"""
gpu.py - GPU-accelerated computations for time series analysis.

This module provides GPU-accelerated implementations of computationally intensive
operations, particularly Dynamic Time Warping (DTW) distance calculations.
It requires CUDA-compatible hardware and the cudtw package to be installed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import cudtw  # https://github.com/***  (Python bindings for CUDA DTW)
except Exception:
    cudtw = None


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D [n_series, length].")
    return X


def cdist_dtw_gpu(
    X: np.ndarray, band: Optional[int] = None, device: int = 0
) -> np.ndarray:
    if cudtw is None:
        raise RuntimeError("cudtw not found. Install cudtw for GPU DTW.")
    X = _ensure_2d(X)
    # cudtw expects C-contiguous arrays on device 0 by default
    # API assumed: cudtw.cdist_dtw(X, band=None, device=0) -> np.ndarray [n,n]
    return np.asarray(cudtw.cdist_dtw(X, band=band, device=device), dtype=float)


def cdist_dtw_gpu_block(
    A: np.ndarray, B: np.ndarray, band: Optional[int] = None, device: int = 0
) -> np.ndarray:
    if cudtw is None:
        raise RuntimeError("cudtw not found. Install cudtw for GPU DTW.")
    A = _ensure_2d(A)
    B = _ensure_2d(B)
    # API assumed: cudtw.cdist_dtw_block(A, B, band=None, device=0) -> np.ndarray [na, nb]
    if hasattr(cudtw, "cdist_dtw_block"):
        return np.asarray(
            cudtw.cdist_dtw_block(A, B, band=band, device=device), dtype=float
        )
    # Fallback: stack and slice with single call
    X = np.vstack([A, B])
    D = np.asarray(cudtw.cdist_dtw(X, band=band, device=device), dtype=float)
    na = A.shape[0]
    nb = B.shape[0]
    return D[:na, na : na + nb]
