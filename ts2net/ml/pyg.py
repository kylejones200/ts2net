"""
PyTorch Geometric adapters for ts2net graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ts2net.core.graph import Graph

if TYPE_CHECKING:
    from torch_geometric.data import Data


def to_pyg_data(
    graph: Graph,
    node_features: NDArray[np.float64] | None = None,
    y: float | int | None = None,
    graph_label: float | int | None = None,
) -> Data:
    """
    Convert a :class:`~ts2net.core.graph.Graph` to a PyG ``Data`` object.

    Parameters
    ----------
    graph : Graph
        ts2net graph (integer node indices recommended).
    node_features : array (n_nodes, n_features), optional
        Node feature matrix. Defaults to in-degree column vector.
    y : float or int, optional
        Graph-level label (stored as ``data.y``).
    graph_label : float or int, optional
        Alias for ``y``.

    Returns
    -------
    torch_geometric.data.Data
        PyG graph with ``x``, ``edge_index``, optional ``edge_attr``, ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net import HVG
    >>> from ts2net.ml import to_pyg_data
    >>> hvg = HVG().build(np.random.randn(50))
    >>> data = to_pyg_data(hvg._graph)  # doctest: +SKIP
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric required. Install with: pip install ts2net[pyg]"
        ) from exc

    src, dst, weights = graph.edges_coo()
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    if node_features is None:
        deg = graph.degree_sequence().astype(np.float64).reshape(-1, 1)
        x_arr = deg
    else:
        x_arr = np.asarray(node_features, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)

    data = Data(
        x=torch.tensor(x_arr, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=graph.n_nodes,
    )

    if weights is not None and len(weights) > 0:
        data.edge_attr = torch.tensor(
            weights.reshape(-1, 1), dtype=torch.float32
        )

    label = y if y is not None else graph_label
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    data.directed = graph.directed
    return data


def panel_to_pyg_list(
    graphs: list[Graph],
    labels: NDArray[Any] | None = None,
    node_features_list: list[NDArray[np.float64]] | None = None,
) -> list[Data]:
    """
    Convert a list of graphs to PyG ``Data`` objects (e.g. for graph classification).

    Parameters
    ----------
    graphs : list of Graph
        One graph per sample.
    labels : array (n_graphs,), optional
        Graph-level labels.
    node_features_list : list of arrays, optional
        Per-graph node features.

    Returns
    -------
    list of Data
    """
    out = []
    for i, g in enumerate(graphs):
        y = None if labels is None else labels[i]
        nf = None if node_features_list is None else node_features_list[i]
        out.append(to_pyg_data(g, node_features=nf, y=y))
    return out
