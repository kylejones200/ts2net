"""
Tests for scale and performance utilities (horizon 0.6).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp

from ts2net import HVG, build_windows
from ts2net.scale import (
    build_windows_streaming,
    edges_to_csr,
    get_performance_contract,
    iter_series_chunks,
    iter_windows,
    list_performance_contracts,
    stream_chunk_stats,
    to_sparse_csr,
)


class TestStreaming:
    def test_iter_windows_count(self):
        x = np.arange(100, dtype=np.float64)
        windows = list(iter_windows(x, width=10, step=5))
        assert len(windows) == 19
        assert windows[0][1] == 0
        assert np.array_equal(windows[0][2], x[0:10])

    def test_iter_series_chunks(self):
        x = np.arange(25, dtype=np.float64)
        chunks = list(iter_series_chunks(x, chunk_size=10, overlap=2))
        assert len(chunks) >= 2
        assert len(chunks[0][1]) == 10

    def test_build_windows_streaming_matches_build_windows(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        ref = build_windows(x, window=40, step=20, method="hvg")
        stream = {
            i: stats
            for i, _, stats in build_windows_streaming(
                x, window=40, step=20, method="hvg"
            )
        }
        for i in range(len(ref["n_nodes"])):
            assert stream[i]["n_nodes"] == ref["n_nodes"][i]
            assert stream[i]["n_edges"] == ref["n_edges"][i]

    def test_build_windows_parallel_matches_serial(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(400)
        serial = build_windows(x, window=30, step=15, method="hvg")
        parallel = build_windows(x, window=30, step=15, method="hvg", n_jobs=2)
        np.testing.assert_array_equal(serial["n_edges"], parallel["n_edges"])

    def test_build_windows_streaming_flag(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(300)
        a = build_windows(x, window=25, step=10, method="hvg", streaming=True)
        b = build_windows(x, window=25, step=10, method="hvg")
        np.testing.assert_array_equal(a["avg_degree"], b["avg_degree"])

    def test_stream_chunk_stats(self):
        x = np.sin(np.linspace(0, 20 * np.pi, 500))
        chunks = list(stream_chunk_stats(x, chunk_size=100, method="hvg"))
        assert len(chunks) == 5
        assert all(stats["n_edges"] > 0 for _, stats in chunks)

    def test_iter_series_chunks_memmap(self, tmp_path):
        path = tmp_path / "series.bin"
        x = np.arange(30, dtype=np.float64)
        x.tofile(path)
        chunks = list(iter_series_chunks(str(path), chunk_size=10))
        assert len(chunks) == 3
        assert len(chunks[0][1]) == 10


class TestParquetStreaming:
    def test_iter_parquet_value_chunks(self, tmp_path):
        pl = pytest.importorskip("polars")
        from ts2net.scale import iter_parquet_value_chunks

        path = tmp_path / "meter.parquet"
        pl.DataFrame({"t": range(25), "v": np.arange(25, dtype=float)}).write_parquet(
            path
        )

        chunks = list(iter_parquet_value_chunks(path, value_col="v", chunk_size=10))
        assert len(chunks) == 3
        assert len(chunks[0][1]) == 10
        assert chunks[0][1][0] == 0.0
        assert chunks[-1][1][-1] == 24.0


class TestSparse:
    def test_edges_to_csr_undirected(self):
        mat = edges_to_csr([(0, 1), (1, 2)], n_nodes=3, directed=False)
        assert isinstance(mat, sp.csr_matrix)
        assert mat.nnz == 4

    def test_to_sparse_csr_from_graph(self):
        g = HVG().build(np.sin(np.linspace(0, 4 * np.pi, 50)))
        mat = to_sparse_csr(g)
        assert mat.shape == (g.n_nodes, g.n_nodes)
        assert mat.nnz > 0


class TestContracts:
    def test_get_hvg_contract(self):
        c = get_performance_contract("hvg")
        assert "O(n)" in c.time_complexity

    def test_list_contracts(self):
        contracts = list_performance_contracts()
        assert "nvg" in contracts
        assert "build_windows" in contracts

    def test_unknown_contract_raises(self):
        with pytest.raises(KeyError):
            get_performance_contract("not_a_method")
