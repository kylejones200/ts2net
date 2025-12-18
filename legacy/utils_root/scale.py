from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Callable, Tuple
from .gpu import cdist_dtw_gpu, cdist_dtw_gpu_block
import os

try:
    import joblib
    from joblib import Parallel, delayed
except Exception:
    joblib = None
    Parallel = None

    def delayed(f):
        return f  # type: ignore


try:
    from dask.distributed import Client, as_completed
except Exception:
    Client = None
    as_completed = None

from .core_rust import cdist_dtw as _cdist_dtw_rs


def dtw_distance_file_gpu(
    X: np.ndarray,
    out_path: str,
    block: int = 256,
    band: int | None = None,
    device: int = 0,
) -> str:
    X = np.asarray(X, float)
    n = X.shape[0]
    D = _ensure_memmap(out_path, (n, n))
    D[:] = np.nan
    D.flush()
    for i0, i1, j0, j1 in _tri_blocks(n, block):
        if i0 == j0:
            B = cdist_dtw_gpu(X[i0:i1], band=band, device=device)
        else:
            B = cdist_dtw_gpu_block(X[i0:i1], X[j0:j1], band=band, device=device)
        _fill_symmetric(D, i0, i1, j0, j1, B.astype(D.dtype, copy=False))
        D.flush()
    return str(Path(out_path))


def _ensure_memmap(
    path: str | os.PathLike, shape: Tuple[int, int], dtype=np.float32
) -> np.memmap:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        fp = np.memmap(p, mode="w+", dtype=dtype, shape=shape)
        fp[:] = np.nan
        fp.flush()
    return np.memmap(p, mode="r+", dtype=dtype, shape=shape)


def _tri_blocks(n: int, block: int):
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            yield i0, i1, j0, j1


def _fill_symmetric(D: np.ndarray, i0: int, i1: int, j0: int, j1: int, B: np.ndarray):
    D[i0:i1, j0:j1] = B
    if j0 != i0:
        D[j0:j1, i0:i1] = B.T


def block_cdist(
    X: np.ndarray,
    kernel: Callable[[np.ndarray, int | None], np.ndarray],
    out_path: str,
    block: int = 256,
    band: int | None = None,
    n_jobs: int = -1,
) -> str:
    X = np.asarray(X, float)
    n = X.shape[0]
    D = _ensure_memmap(out_path, (n, n))
    D[:] = np.nan
    D.flush()

    if Parallel is None:
        for i0, i1, j0, j1 in _tri_blocks(n, block):
            B = (
                kernel(X[i0:i1], band=None)
                if i0 == j0
                else _cdist_block(kernel, X[i0:i1], X[j0:j1], band)
            )
            _fill_symmetric(D, i0, i1, j0, j1, B.astype(D.dtype, copy=False))
            D.flush()
        return str(Path(out_path))

    def job(i0, i1, j0, j1):
        if i0 == j0:
            B = kernel(X[i0:i1], band=band)
        else:
            B = _cdist_block(kernel, X[i0:i1], X[j0:j1], band)
        return (i0, i1, j0, j1, B.astype(np.float32, copy=False))

    tasks = list(_tri_blocks(n, block))
    for i0, i1, j0, j1, B in Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(job)(*t) for t in tasks
    ):
        _fill_symmetric(D, i0, i1, j0, j1, B)
    D.flush()
    return str(Path(out_path))


def _cdist_block(kernel, A: np.ndarray, B: np.ndarray, band: int | None):
    # reuse the same kernel by stacking
    X = np.vstack([A, B])
    D = kernel(X, band=band)
    na = A.shape[0]
    nb = B.shape[0]
    return D[:na, na : na + nb]


def dtw_distance_file(
    X: np.ndarray,
    out_path: str,
    block: int = 256,
    band: int | None = None,
    n_jobs: int = -1,
) -> str:
    return block_cdist(
        X, _cdist_dtw_rs, out_path, block=block, band=band, n_jobs=n_jobs
    )


def dtw_distance_dask(
    X: np.ndarray,
    out_path: str,
    block: int = 256,
    band: int | None = None,
    client=None,
) -> str:
    if client is None:
        if Client is None:
            raise RuntimeError("dask.distributed not available")
        client = Client()
    X = np.asarray(X, float)
    n = X.shape[0]
    D = _ensure_memmap(out_path, (n, n))
    D[:] = np.nan
    D.flush()

    futures = {}
    for t in _tri_blocks(n, block):
        i0, i1, j0, j1 = t
        if i0 == j0:
            fut = client.submit(_cdist_dtw_rs, X[i0:i1], band)
        else:
            fut = client.submit(_cdist_block, _cdist_dtw_rs, X[i0:i1], X[j0:j1], band)
        futures[fut] = t

    for fut in as_completed(futures):
        i0, i1, j0, j1 = futures[fut]
        B = fut.result()
        _fill_symmetric(D, i0, i1, j0, j1, B.astype(np.float32, copy=False))
        D.flush()

    return str(Path(out_path))
