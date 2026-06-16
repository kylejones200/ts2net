"""
Sparse matrix helpers for large graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse as sp

if TYPE_CHECKING:
    from ts2net.core.graph import Graph


def to_sparse_csr(
    graph: Graph,
    dtype: type = np.float64,
) -> sp.csr_matrix:
    """
    Return a CSR adjacency matrix without densifying.

    Parameters
    ----------
    graph : Graph
        ts2net graph result.
    dtype : numpy dtype, default float64
        Matrix value dtype.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency of shape (n_nodes, n_nodes).
    """
    adj = graph.adjacency_matrix(format="sparse")
    if graph.weighted:
        return adj.astype(dtype)
    return adj.astype(dtype)


def edges_to_csr(
    edges: list[tuple],
    n_nodes: int,
    directed: bool = False,
    weighted: bool = False,
    dtype: type = np.float64,
) -> sp.csr_matrix:
    """
    Build a CSR matrix directly from an edge list.

    Avoids constructing a dense adjacency for large sparse graphs.
    """
    if not edges:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=dtype)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        w = float(edge[2]) if weighted and len(edge) > 2 else 1.0
        rows.append(u)
        cols.append(v)
        data.append(w)
        if not directed and u != v:
            rows.append(v)
            cols.append(u)
            data.append(w)

    coo = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    return coo.tocsr()
