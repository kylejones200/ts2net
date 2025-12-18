"""
Utility functions for parity testing between R and Python implementations.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple
import numpy as np
import networkx as nx

from .models import ParityCase


def load_series(path: str) -> np.ndarray:
    """Load a time series from a CSV file."""
    import pandas as pd
    return pd.read_csv(path, header=None).squeeze("columns").to_numpy(float)


def read_r_graphml(path: str) -> nx.Graph:
    """Read a GraphML file created by R's igraph."""
    return nx.read_graphml(path)


def compare_graphs(Gr: nx.Graph, Gp: nx.Graph) -> Tuple[float, float, float, float, float]:
    """Compare two graphs and return similarity metrics.
    
    Returns:
        Tuple containing (jaccard, deg_l1, tri_rel, C_rel, L_rel)
    """
    Hr = Gr.to_undirected()
    Hp = Gp.to_undirected()
    Er = {tuple(sorted(e)) for e in Hr.edges()}
    Ep = {tuple(sorted(e)) for e in Hp.edges()}
    inter = len(Er & Ep)
    union = len(Er | Ep) or 1
    jacc = inter / union
    
    # Degree L1 per node after union set of nodes
    nodes = sorted(set(Hr.nodes()) | set(Hp.nodes()))
    dr = np.array([Hr.degree(n) if n in Hr else 0 for n in nodes], float)
    dp = np.array([Hp.degree(n) if n in Hp else 0 for n in nodes], float)
    deg_l1 = float(np.mean(np.abs(dr - dp))) if nodes else 0.0
    
    # Triangles and small-world
    tri_r = sum(nx.triangles(Hr).values()) // 3 if Hr.number_of_nodes() else 0
    tri_p = sum(nx.triangles(Hp).values()) // 3 if Hp.number_of_nodes() else 0
    tri_rel = 0.0 if max(1, tri_r) == 0 else abs(tri_p - tri_r) / max(1, tri_r)
    
    # Small-world coefficients
    def rel(a, b):
        if (
            (a is None)
            or (b is None)
            or (not math.isfinite(a))
            or (not math.isfinite(b))
            or a == 0
        ):
            return float("nan")
        return abs(b - a) / abs(a)
    
    # Get graph summaries
    from ts2net.core import graph_summary
    sr = graph_summary(Hr)._raw
    sp = graph_summary(Hp)._raw
    
    return jacc, deg_l1, tri_rel, rel(sr["C"], sp["C"]), rel(sr["L"], sp["L"])


def nx_from_python_case(case: ParityCase) -> nx.Graph:
    """Create a NetworkX graph from a ParityCase using the Python implementation."""
    from ts2net.core.visibility import HVG, NVG
    from ts2net.core.recurrence import RecurrenceNetwork
    from ts2net.core.transition import TransitionNetwork
    
    if case.kind == "HVG":
        x = load_series(case.series)
        G, _ = HVG(backend="rust", sparse=False).fit_transform(x)
        return G
    if case.kind == "NVG":
        x = load_series(case.series)
        G, _ = NVG(backend="rust", sparse=False).fit_transform(x)
        return G
    if case.kind == "RN":
        x = load_series(case.series)
        p = case.params or {}
        rn = RecurrenceNetwork(
            m=p.get("m", 2),
            tau=p.get("tau", 1),
            rule=p.get("rule", "epsilon"),
            epsilon=p.get("epsilon"),
            k=p.get("k", 10),
            theiler=p.get("theiler", 0),
            metric=p.get("metric", "euclidean"),
            sparse=False,
        )
        G, _ = rn.fit_transform(x)
        return G
    if case.kind == "TN":
        x = load_series(case.series)
        p = case.params or {}
        tn = TransitionNetwork(
            symbolizer=p.get("symbolizer", "ordinal"),
            order=p.get("order", 3),
            delay=p.get("delay", 1),
            tie_rule=p.get("tie_rule", "stable"),
            bins=p.get("bins", 5),
        )
        G, _ = tn.fit_transform(x)
        return G
    raise ValueError(f"Unknown graph kind: {case.kind}")
