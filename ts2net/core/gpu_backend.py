"""
GPU array backend detection and selection (Horizon 4 / v0.6).
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

GpuBackend = Literal["auto", "torch", "cupy", "cpu"]
DeviceKind = Literal["auto", "cpu", "cuda", "gpu"]


def cupy_available() -> bool:
    """Return True when CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def torch_available() -> bool:
    """Return True when PyTorch is importable."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    """Return True when a CUDA device is usable (PyTorch or CuPy)."""
    if torch_available():
        import torch

        if torch.cuda.is_available():
            return True
    if cupy_available():
        try:
            import cupy as cp

            return cp.cuda.is_available()
        except Exception:
            return False
    return False


def resolve_gpu_backend(
    requested: GpuBackend | str = "auto",
    *,
    device: DeviceKind | str = "auto",
    allow_fallback: bool = True,
) -> str:
    """
    Resolve a GPU compute backend to ``torch``, ``cupy``, or ``cpu``.

    ``TS2NET_GPU_BACKEND`` overrides ``auto`` when set to ``torch`` or ``cupy``.
    """
    req = (requested or "auto").lower()
    if req not in ("auto", "torch", "cupy", "cpu"):
        raise ValueError(f"gpu backend must be auto|torch|cupy|cpu, got {req!r}")

    dev = (device or "auto").lower()
    want_gpu = dev in ("gpu", "cuda", "auto") and (
        dev in ("gpu", "cuda") or os.environ.get("TS2NET_DEVICE", "").lower() in ("gpu", "cuda")
    )
    if dev == "cpu":
        return "cpu"

    if req == "auto":
        env = os.environ.get("TS2NET_GPU_BACKEND", "").lower()
        if env in ("torch", "cupy", "cpu"):
            req = env
        elif want_gpu or os.environ.get("TS2NET_DEVICE", "").lower() in ("gpu", "cuda"):
            req = "auto"

    if req == "cpu":
        return "cpu"

    if req == "torch" or (req == "auto" and torch_available()):
        if torch_available():
            if want_gpu and not cuda_available():
                if allow_fallback:
                    warnings.warn(
                        "GPU requested but CUDA unavailable; using torch CPU.",
                        stacklevel=2,
                    )
                else:
                    raise RuntimeError("CUDA requested but unavailable")
            return "torch"
        if req == "torch" and not allow_fallback:
            raise ImportError("torch backend requested but torch is not installed")

    if req == "cupy" or (req == "auto" and cupy_available()):
        if cupy_available():
            if want_gpu and not cuda_available():
                if allow_fallback:
                    warnings.warn(
                        "GPU requested but CUDA unavailable; using cupy CPU path.",
                        stacklevel=2,
                    )
                else:
                    raise RuntimeError("CUDA requested but unavailable")
            return "cupy"
        if req == "cupy" and not allow_fallback:
            raise ImportError("cupy backend requested but cupy is not installed")

    return "cpu"


def resolve_torch_device(device: DeviceKind | str = "auto") -> str:
    """Map device kind to a torch device string."""
    dev = (device or "auto").lower()
    env = os.environ.get("TS2NET_DEVICE", "").lower()
    if dev in ("gpu", "cuda") or env in ("gpu", "cuda"):
        if torch_available():
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    if dev == "cpu":
        return "cpu"
    if dev == "auto" and env == "cpu":
        return "cpu"
    if torch_available():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"
