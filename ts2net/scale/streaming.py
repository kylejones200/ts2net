"""
Streaming and chunked processing for large time series.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_positive_int, validate_series
from ..factory import aggregate_stats, create_graph_builder
from .contracts import get_performance_contract


def iter_windows(
    x: NDArray[np.float64],
    width: int,
    step: int = 1,
    start: int = 0,
    end: int | None = None,
    copy: bool = False,
) -> Iterator[tuple[int, int, NDArray[np.float64]]]:
    """
    Yield sliding windows without materializing the full window matrix.

    Parameters
    ----------
    x : array (n_points,)
        Input time series.
    width : int
        Window width.
    step : int, default 1
        Step between consecutive windows.
    start, end : int, optional
        Slice bounds on ``x`` (same semantics as ``ts_to_windows``).
    copy : bool, default False
        If True, copy each window slice (safer if ``x`` is mutated).

    Yields
    ------
    window_index : int
        Zero-based window index.
    start_index : int
        Start position in ``x``.
    window : array (width,)
        Window values.
    """
    x = validate_series(x, "iter_windows")
    width = validate_positive_int("width", width)
    step = validate_positive_int("step", step)

    n = len(x)
    if end is None:
        end = n
    if width > (end - start):
        raise ValueError(f"width ({width}) cannot exceed series length ({end - start})")

    n_windows = (end - start - width) // step + 1
    if n_windows <= 0:
        raise ValueError(
            f"No windows possible with width={width}, step={step}, "
            f"start={start}, end={end}"
        )

    for i in range(n_windows):
        w_start = start + i * step
        w_end = w_start + width
        window = x[w_start:w_end]
        if copy:
            window = window.copy()
        yield i, w_start, window


def iter_series_chunks(
    x: NDArray[np.float64] | str | Path,
    chunk_size: int,
    overlap: int = 0,
    dtype: type = np.float64,
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """
    Iterate over contiguous chunks of a series or memory-mapped file.

    Parameters
    ----------
    x : array or path
        In-memory series or path to a binary/memmap file.
    chunk_size : int
        Points per chunk.
    overlap : int, default 0
        Overlap between consecutive chunks.
    dtype : numpy dtype, default float64
        Dtype when loading from file.

    Yields
    ------
    chunk_index : int
    chunk : array
    """
    chunk_size = validate_positive_int("chunk_size", chunk_size)
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if isinstance(x, (str, Path)):
        arr = np.memmap(x, dtype=dtype, mode="r")
    else:
        arr = validate_series(x, "iter_series_chunks")

    stride = chunk_size - overlap
    n = len(arr)
    idx = 0
    chunk_i = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        chunk = np.asarray(arr[idx:end], dtype=np.float64)
        yield chunk_i, chunk
        if end >= n:
            break
        idx += stride
        chunk_i += 1


def _make_window_config(method: str, window: int, output: str, method_kwargs: dict):
    from ..config import HVGConfig, NVGConfig, RecurrenceConfig, TransitionConfig

    method = method.lower()
    if method == "hvg":
        return HVGConfig(
            enabled=True,
            output=output,
            weighted=method_kwargs.get("weighted", False),
            limit=method_kwargs.get("limit"),
            directed=method_kwargs.get("directed", False),
        )
    if method == "nvg":
        return NVGConfig(
            enabled=True,
            output=output,
            weighted=method_kwargs.get("weighted", False),
            limit=method_kwargs.get("limit", min(100, window)),
            max_edges=method_kwargs.get("max_edges"),
            max_edges_per_node=method_kwargs.get("max_edges_per_node"),
            max_memory_mb=method_kwargs.get("max_memory_mb"),
        )
    if method == "recurrence":
        return RecurrenceConfig(
            enabled=True,
            output=output,
            m=method_kwargs.get("m", 3),
            rule="knn",
            k=method_kwargs.get("k", 5),
            tau=method_kwargs.get("tau", 1),
            epsilon=method_kwargs.get("epsilon", 0.1),
            metric=method_kwargs.get("metric", "euclidean"),
        )
    if method == "transition":
        return TransitionConfig(
            enabled=True,
            output=output,
            symbolizer=method_kwargs.get("symbolizer", "ordinal"),
            order=method_kwargs.get("order", 3),
            n_states=method_kwargs.get("n_states"),
        )
    raise ValueError(
        f"Unknown method: {method}. Must be one of hvg, nvg, recurrence, transition"
    )


def _stats_for_window(
    window_data: NDArray[np.float64],
    method: str,
    config: Any,
) -> dict[str, float]:
    builder = create_graph_builder(method, config, n_points=len(window_data))
    builder.build(window_data)
    return builder.stats()


def build_windows_streaming(
    x: NDArray[np.float64],
    window: int,
    step: int = 1,
    method: str = "hvg",
    output: str = "stats",
    aggregate: str | None = None,
    **method_kwargs,
) -> Iterator[tuple[int, int, dict[str, float] | float]]:
    """
    Stream graph statistics per window (constant memory in window count).

    Unlike :func:`ts2net.api_windows.build_windows`, this does not allocate
    an ``(n_windows, window)`` matrix.

    Yields
    ------
    window_index : int
    start_index : int
    stats : dict or float
        Per-window stats, or a single aggregated value if ``aggregate`` is set.
    """
    config = _make_window_config(method, window, output, method_kwargs)
    for i, start_idx, window_data in iter_windows(x, window, step):
        try:
            stats = _stats_for_window(window_data, method.lower(), config)
            if aggregate:
                yield i, start_idx, aggregate_stats(stats, aggregate)
            else:
                yield i, start_idx, stats
        except Exception:
            if aggregate:
                yield i, start_idx, float("nan")
            else:
                yield i, start_idx, _empty_stats()


def _empty_stats() -> dict[str, float]:
    return {
        "n_nodes": 0,
        "n_edges": 0,
        "avg_degree": float("nan"),
        "std_degree": float("nan"),
    }


def estimate_window_job_memory_mb(window: int, method: str = "hvg") -> float:
    """
    Rough per-window memory estimate in megabytes.

    Uses performance contracts as guidance for logging and capacity planning.
    """
    contract = get_performance_contract(method)
    _ = contract
    base = window * 8 / (1024 * 1024)
    if method.lower() == "recurrence":
        return base * window
    if method.lower() == "nvg":
        return base * 4
    return base


def iter_parquet_value_chunks(
    path: str | Path,
    value_col: str,
    chunk_size: int = 50_000,
    time_col: str | None = None,
    id_col: str | None = None,
    series_id: str | None = None,
) -> Iterator[tuple[int, NDArray[np.float64]]]:
    """
    Yield value chunks from a Parquet file without loading the full series.

    Requires the optional ``polars`` package (``uv sync --extra polars``).

    Parameters
    ----------
    path : str or Path
        Parquet file or directory.
    value_col : str
        Numeric value column.
    chunk_size : int, default 50_000
        Rows per yielded chunk.
    time_col : str, optional
        Sort by this column before chunking.
    id_col : str, optional
        Filter to a single series id when set with ``series_id``.
    series_id : str, optional
        Series identifier value when ``id_col`` is provided.

    Yields
    ------
    chunk_index : int
    values : array
        Chunk of ``value_col`` as float64.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "iter_parquet_value_chunks requires polars. "
            "Install with: uv sync --extra polars"
        ) from exc

    chunk_size = validate_positive_int("chunk_size", chunk_size)
    lf = pl.scan_parquet(str(path))
    if id_col is not None:
        if series_id is None:
            raise ValueError("series_id is required when id_col is set")
        lf = lf.filter(pl.col(id_col) == series_id)
    if time_col is not None:
        lf = lf.sort(time_col)

    n_rows = int(lf.select(pl.len()).collect().item())
    if n_rows == 0:
        return

    for chunk_i, offset in enumerate(range(0, n_rows, chunk_size)):
        chunk = (
            lf.slice(offset, chunk_size)
            .select(value_col)
            .collect()
            .get_column(value_col)
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        yield chunk_i, chunk


def stream_chunk_stats(
    source: NDArray[np.float64] | str | Path,
    chunk_size: int,
    method: str = "hvg",
    overlap: int = 0,
    **method_kwargs,
) -> Iterator[tuple[int, dict[str, float]]]:
    """
    Compute graph summary stats on contiguous chunks (memmap-safe).

    Parameters
    ----------
    source : array or path
        In-memory series or memmap file path.
    chunk_size : int
        Points per chunk.
    method : str, default "hvg"
        Builder name: ``hvg``, ``nvg``, ``recurrence``, ``transition``.
    overlap : int, default 0
        Overlap between chunks (see :func:`iter_series_chunks`).
    **method_kwargs
        Passed to the graph builder config.

    Yields
    ------
    chunk_index : int
    stats : dict
        Graph summary for the chunk.
    """
    config = _make_window_config(method, chunk_size, "stats", method_kwargs)
    method_key = method.lower()

    for chunk_i, chunk in iter_series_chunks(source, chunk_size, overlap=overlap):
        if len(chunk) < 2:
            yield chunk_i, {
                "n_nodes": len(chunk),
                "n_edges": 0,
                "avg_degree": 0.0,
                "std_degree": 0.0,
            }
            continue
        try:
            yield chunk_i, _stats_for_window(chunk, method_key, config)
        except Exception:
            yield chunk_i, _empty_stats()
