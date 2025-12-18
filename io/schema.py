from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Tuple
import numpy as np
import pandas as pd
import networkx as nx

SCHEMA_VERSION = "1.0"


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def graph_to_parquet(
    G: nx.Graph, base_path: str, graph_name: str = "default"
) -> Tuple[str, str, str]:
    p = Path(base_path)
    p.mkdir(parents=True, exist_ok=True)
    directed = G.is_directed()
    nodes = []
    for n, d in G.nodes(data=True):
        nodes.append(
            {
                "id": str(n),
                "label": str(d.get("label", n)),
                "attrs": json.dumps(
                    {k: _to_jsonable(v) for k, v in d.items() if k != "label"}
                ),
            }
        )
    edges = []
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        attrs = {k: _to_jsonable(vv) for k, vv in d.items() if k != "weight"}
        edges.append(
            {
                "src": str(u),
                "dst": str(v),
                "weight": w,
                "directed": bool(directed),
                "attrs": json.dumps(attrs),
            }
        )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "graph_name": graph_name,
        "directed": bool(directed),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
    }
    nodes_df = pd.DataFrame(nodes, columns=["id", "label", "attrs"])
    edges_df = pd.DataFrame(
        edges, columns=["src", "dst", "weight", "directed", "attrs"]
    )
    nodes_path = str(p / "nodes.parquet")
    edges_path = str(p / "edges.parquet")
    meta_path = str(p / "meta.json")
    nodes_df.to_parquet(nodes_path, index=False)
    edges_df.to_parquet(edges_path, index=False)
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    return nodes_path, edges_path, meta_path


def parquet_to_graph(base_path: str) -> nx.Graph:
    p = Path(base_path)
    meta = json.loads((p / "meta.json").read_text())
    directed = bool(meta.get("directed", False))
    G = nx.DiGraph() if directed else nx.Graph()
    nodes_df = pd.read_parquet(p / "nodes.parquet")
    edges_df = pd.read_parquet(p / "edges.parquet")
    for _, row in nodes_df.iterrows():
        attrs = {}
        if isinstance(row["attrs"], str) and row["attrs"]:
            attrs = json.loads(row["attrs"])
        label = row.get("label", row["id"])
        G.add_node(row["id"], label=label, **attrs)
    for _, row in edges_df.iterrows():
        attrs = {}
        if isinstance(row["attrs"], str) and row["attrs"]:
            attrs = json.loads(row["attrs"])
        G.add_edge(row["src"], row["dst"], weight=float(row["weight"]), **attrs)
    return G


def write_graphml(G: nx.Graph, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, out)
    return str(out)


def write_gexf(G: nx.Graph, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, out)
    return str(out)
