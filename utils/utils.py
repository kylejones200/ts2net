"""
utils.py - Utility functions for graph operations and data export.

This module provides various utility functions for working with graphs, including
exporting to different file formats, parameter handling for scikit-learn compatibility,
and other helper functions.
"""

from __future__ import annotations
from .schema import graph_to_parquet, parquet_to_graph, write_graphml, write_gexf
import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd


def export_edgelist_parquet(G: nx.Graph, path: str) -> str:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append((u, v, d.get("weight", 1.0)))
    df = pd.DataFrame(rows, columns=["u", "v", "weight"])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return str(p)


def export_graph(G: nx.Graph, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".graphml":
        nx.write_graphml(G, p)
    elif ext == ".gexf":
        nx.write_gexf(G, p)
    elif ext in (".edgelist", ".csv"):
        with p.open("w") as f:
            for u, v, d in G.edges(data=True):
                w = d.get("weight", 1.0)
                f.write(f"{u},{v},{w}\n")
    elif ext == ".json":
        data = nx.node_link_data(G)
        p.write_text(json.dumps(data))
    else:
        raise ValueError("Use .graphml, .gexf, .edgelist, .csv, or .json")
    return str(p)


def export_adj(A, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".npy":
        import numpy as np

        np.save(p, A)
    elif ext == ".npz":
        try:
            from scipy.sparse import issparse, save_npz
        except Exception:
            raise RuntimeError("SciPy required for .npz")
        if not issparse(A):
            raise ValueError("Use .npy for dense arrays")
        save_npz(p, A)
    else:
        raise ValueError("Use .npy for dense or .npz for CSR")
    return str(p)


class SKMixin:
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dict__ if not k.endswith("_")}

    def set_params(self, **params) -> "SKMixin":
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter: {k}")
            setattr(self, k, v)
        return self
