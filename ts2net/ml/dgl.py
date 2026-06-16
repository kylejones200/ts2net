"""
DGL adapters for ts2net graphs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ts2net.core.graph import Graph

if TYPE_CHECKING:
    import dgl


def to_dgl_graph(
    graph: Graph,
    node_features: NDArray[np.float64] | None = None,
    feature_name: str = "feat",
    weight_name: str = "weight",
) -> dgl.DGLGraph:
    """
    Convert a :class:`~ts2net.core.graph.Graph` to a DGL graph.

    Parameters
    ----------
    graph : Graph
        ts2net graph (integer node indices recommended).
    node_features : array (n_nodes, n_features), optional
        Stored in ``g.ndata[feature_name]``. Defaults to degree features.
    feature_name : str, default "feat"
        Node data field name.
    weight_name : str, default "weight"
        Edge weight field name.

    Returns
    -------
    dgl.DGLGraph

    Examples
    --------
    >>> import numpy as np
    >>> from ts2net import HVG
    >>> from ts2net.ml import to_dgl_graph
    >>> hvg = HVG().build(np.random.randn(50))
    >>> g = to_dgl_graph(hvg._graph)  # doctest: +SKIP
    """
    try:
        import dgl
        import torch
    except ImportError as exc:
        raise ImportError(
            "DGL required. Install with: pip install ts2net[dgl]"
        ) from exc

    src, dst, weights = graph.edges_coo()
    g = dgl.graph(
        (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)),
        num_nodes=graph.n_nodes,
    )

    if weights is not None and len(weights) > 0:
        g.edata[weight_name] = torch.tensor(weights, dtype=torch.float32)

    if node_features is None:
        feat = graph.degree_sequence().astype(np.float32).reshape(-1, 1)
    else:
        feat = np.asarray(node_features, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat.reshape(-1, 1)

    g.ndata[feature_name] = torch.tensor(feat, dtype=torch.float32)
    return g


def panel_to_dgl_list(
    graphs: list[Graph],
    labels: NDArray[Any] | None = None,
    node_features_list: list[NDArray[np.float64]] | None = None,
) -> tuple[list, Any | None]:
    """
    Convert a panel of graphs to DGL graphs and optional label tensor.

    Returns
    -------
    graphs : list of DGLGraph
    labels : torch.Tensor or None
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required for DGL labels tensor") from exc

    dgl_graphs = []
    for i, gr in enumerate(graphs):
        nf = None if node_features_list is None else node_features_list[i]
        dgl_graphs.append(to_dgl_graph(gr, node_features=nf))

    y_tensor = None
    if labels is not None:
        y_tensor = torch.tensor(labels, dtype=torch.long)
    return dgl_graphs, y_tensor
