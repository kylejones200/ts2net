"""
Dask and Ray execution helpers for scale workloads (Horizon 4 / v0.6).

These run on the local CPU by default. GPU is optional and only used when
``ts_dist(..., device='gpu')`` is set explicitly.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

ExecutorKind = Literal["joblib", "dask", "ray"]
T = TypeVar("T")

# Keep Ray local runs lightweight (no GPU, no huge data uploads).
_RAY_RUNTIME_EXCLUDES = [
    ".git",
    ".venv",
    "experiments",
    "benchmarks/results",
    "htmlcov",
    ".pytest_cache",
    "**/*.csv",
    "**/*.npz",
    "uv.lock",
]


def _init_ray(n_workers: int | None) -> None:
    import ray

    if ray.is_initialized():
        return
    ray.init(
        num_cpus=n_workers or os.cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={"excludes": _RAY_RUNTIME_EXCLUDES},
    )


def _ray_ts_dist_part(
    arr: NDArray[np.float64],
    start: int,
    end: int,
    method: str,
    kwargs: dict[str, Any],
) -> NDArray[np.float64]:
    from ..multivariate.distances import ts_dist_part

    return ts_dist_part(arr, start, end, method=method, **kwargs)


def _ray_window_stats(
    window: NDArray[np.float64],
    method_key: str,
    config: object,
) -> dict[str, float]:
    from ..api_windows import _compute_window_stats

    return _compute_window_stats(window, method_key, config, None)


def dask_available() -> bool:
    try:
        import dask  # noqa: F401

        return True
    except ImportError:
        return False


def ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def parallel_map(
    func: Callable[..., T],
    items: list[tuple[Any, ...]],
    *,
    executor: ExecutorKind = "dask",
    n_workers: int | None = None,
) -> list[T]:
    """
    Map ``func(*args)`` over ``items`` using Dask, Ray, or joblib.

    For Ray, prefer the dedicated helpers :func:`ts_dist_distributed` and
    :func:`build_windows_distributed` (module-level workers, faster local startup).
    """
    if not items:
        return []

    if executor == "joblib":
        from joblib import Parallel, delayed

        n_jobs = n_workers if n_workers is not None else -1
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(func)(*args) for args in items
        )

    if executor == "dask":
        if not dask_available():
            raise ImportError(
                "Dask executor requested but dask is not installed. "
                "Install with: pip install 'ts2net[distributed]'"
            )
        from dask import compute, delayed

        tasks = [delayed(func)(*args) for args in items]
        if n_workers is not None:
            from dask.distributed import Client, LocalCluster

            with LocalCluster(n_workers=n_workers, threads_per_worker=1) as cluster:
                with Client(cluster):
                    return list(compute(*tasks))
        return list(compute(*tasks))

    if executor == "ray":
        if not ray_available():
            raise ImportError(
                "Ray executor requested but ray is not installed. "
                "Install with: pip install 'ts2net[distributed]'"
            )
        import ray

        _init_ray(n_workers)
        remote = ray.remote(func)
        futures = [remote.remote(*args) for args in items]
        return ray.get(futures)

    raise ValueError(f"Unknown executor {executor!r}. Use joblib, dask, or ray.")


def ts_dist_distributed(
    X: NDArray[np.float64],
    method: str = "correlation",
    *,
    executor: ExecutorKind = "dask",
    row_chunk_size: int = 32,
    n_workers: int | None = None,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """
    Distributed pairwise distance matrix via row blocks of :func:`ts_dist_part`.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if executor == "ray":
        import ray

        _init_ray(n_workers)
        remote = ray.remote(_ray_ts_dist_part)
        kwargs_dict = dict(kwargs)
        futures = [
            remote.remote(X, start, min(start + row_chunk_size, n), method, kwargs_dict)
            for start in range(0, n, row_chunk_size)
        ]
        return np.vstack(ray.get(futures))

    chunks: list[tuple[Any, ...]] = []
    for start in range(0, n, row_chunk_size):
        end = min(start + row_chunk_size, n)
        chunks.append((X, start, end, method))

    from ..multivariate.distances import ts_dist_part

    parts = parallel_map(
        lambda arr, s, e, m: ts_dist_part(arr, s, e, method=m, **kwargs),
        chunks,
        executor=executor if executor != "ray" else "dask",
        n_workers=n_workers,
    )
    return np.vstack(parts)


def build_windows_distributed(
    x: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: str = "hvg",
    *,
    executor: ExecutorKind = "dask",
    n_workers: int | None = None,
    **method_kwargs: Any,
) -> dict[str, NDArray[np.float64]]:
    """
    Build per-window graph stats using a Dask/Ray executor.
    """
    from ..scale.streaming import _make_window_config, iter_windows

    method_key = method.lower()
    config = _make_window_config(method_key, window, "stats", method_kwargs)
    window_list = [w for _, _, w in iter_windows(x, window, step)]

    if executor == "ray":
        import ray

        _init_ray(n_workers)
        remote = ray.remote(_ray_window_stats)
        computed = ray.get(
            [remote.remote(w, method_key, config) for w in window_list]
        )
    else:
        computed = parallel_map(
            lambda w: _ray_window_stats(w, method_key, config),
            [(w,) for w in window_list],
            executor=executor,
            n_workers=n_workers,
        )

    n_windows = len(computed)
    result = {
        "n_nodes": np.zeros(n_windows, dtype=np.int64),
        "n_edges": np.zeros(n_windows, dtype=np.int64),
        "avg_degree": np.zeros(n_windows, dtype=np.float64),
        "std_degree": np.zeros(n_windows, dtype=np.float64),
    }
    for i, stats in enumerate(computed):
        result["n_nodes"][i] = int(stats["n_nodes"])
        result["n_edges"][i] = int(stats["n_edges"])
        result["avg_degree"][i] = float(stats["avg_degree"])
        result["std_degree"][i] = float(stats.get("std_degree", 0.0))
    return result
