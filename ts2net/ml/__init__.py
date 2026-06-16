"""
Graph ML adapters (PyTorch Geometric, DGL).
"""

from .dgl import panel_to_dgl_list, to_dgl_graph
from .pyg import panel_to_pyg_list, to_pyg_data

__all__ = [
    "to_pyg_data",
    "panel_to_pyg_list",
    "to_dgl_graph",
    "panel_to_dgl_list",
]
