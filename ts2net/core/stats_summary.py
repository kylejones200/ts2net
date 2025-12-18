"""
stats_summary1.py - Statistical summary utilities for network and time series analysis.

This module provides classes and functions for generating formatted summaries of
statistical analyses, including correlation tests, graph metrics, and spatial statistics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .summary import SummaryTable, _fmt
from typing import List, Any, Optional, Tuple, Dict


@dataclass
class CorrSigResult:
    r: float
    n: int
    ci: Tuple[float, float]
    alpha: float
    sig: bool

    def summary(self) -> SummaryTable:
        rows = [
            ["r", _fmt(self.r)],
            ["n", str(self.n)],
            [f"CI {1-self.alpha:.2f}", f"[{_fmt(self.ci[0])}, {_fmt(self.ci[1])}]"],
            ["significant", "yes" if self.sig else "no"],
        ]
        return SummaryTable("Correlation test", ["stat", "value"], rows)


@dataclass
class CCFResult:
    r: float
    lag: int
    n_eff: int
    ci: Tuple[float, float]
    alpha: float
    sig: bool

    def summary(self) -> SummaryTable:
        rows = [
            ["max r", _fmt(self.r)],
            ["lag", str(self.lag)],
            ["n_eff", str(self.n_eff)],
            [f"CI {1-self.alpha:.2f}", f"[{_fmt(self.ci[0])}, {_fmt(self.ci[1])}]"],
            ["significant", "yes" if self.sig else "no"],
        ]
        return SummaryTable(
            "Cross-correlation (max over lags)", ["stat", "value"], rows
        )


@dataclass
class EventSyncResult:
    n1: int
    n2: int
    c12: float
    c21: float
    ties: float
    q12: float
    q21: float
    Q: float

    def summary(self) -> SummaryTable:
        rows = [
            ["events x", str(self.n1)],
            ["events y", str(self.n2)],
            ["c12", _fmt(self.c12)],
            ["c21", _fmt(self.c21)],
            ["ties", _fmt(self.ties)],
            ["q12", _fmt(self.q12)],
            ["q21", _fmt(self.q21)],
            ["Q", _fmt(self.Q)],
        ]
        return SummaryTable("Event synchronization", ["stat", "value"], rows)


@dataclass
class GraphSummaryResult:
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    motif_counts: Optional[Dict[str, Any]] = None
    # Legacy field aliases for backward compatibility
    n: Optional[int] = None
    m: Optional[int] = None
    deg_mean: float = 0.0
    deg_std: float = 0.0
    assortativity: float = 0.0
    avg_clustering: float = 0.0
    C: float = 0.0
    L: float = 0.0
    C_er: float = 0.0
    L_er: float = 0.0
    sigma: float = 0.0
    triangles: int = 0
    wedges: int = 0
    other_3_node: int = 0
    N3: int = 0
    
    def __post_init__(self):
        """Handle field aliases and defaults."""
        if self.motif_counts is None:
            self.motif_counts = {}
        # Handle legacy aliases
        if self.n is None:
            self.n = self.n_nodes
        if self.m is None:
            self.m = self.n_edges

    def summary(self) -> SummaryTable:
        rows = [
            ["nodes", str(self.n)],
            ["edges", str(self.m)],
            ["deg_mean", _fmt(self.deg_mean)],
            ["deg_std", _fmt(self.deg_std)],
            ["assortativity", _fmt(self.assortativity)],
            ["avg_clustering", _fmt(self.avg_clustering)],
            ["C", _fmt(self.C)],
            ["L", _fmt(self.L)],
            ["C_er", _fmt(self.C_er)],
            ["L_er", _fmt(self.L_er)],
            ["sigma", _fmt(self.sigma)],
            ["triangles", str(self.triangles)],
            ["wedges", str(self.wedges)],
            ["other_3_node", str(self.other_3_node)],
            ["N3", str(self.N3)],
        ]
        return SummaryTable("Graph summary", ["metric", "value"], rows)


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "nan"
        return f"{x:.{digits}f}"
    return str(x)


def _col_widths(rows: List[List[str]]) -> List[int]:
    n = max(len(r) for r in rows)
    widths = [0] * n
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))
    return widths


def _draw_table(
    title: str, headers: List[str], rows: List[List[str]], foot: Optional[str] = None
) -> str:
    H = [headers] + rows
    widths = _col_widths(H)

    def line(sep_left="+", sep_mid="+", sep_right="+", pad="-"):
        parts = [sep_left]
        for w in widths:
            parts.append(pad * (w + 2))
            parts.append(sep_mid)
        parts[-1] = sep_right
        return "".join(parts)

    def fmt_row(vals: List[str]) -> str:
        cells = []
        for v, w in zip(vals, widths):
            cells.append(" " + v.ljust(w) + " ")
        return "|" + "|".join(cells) + "|"

    out = []
    if title:
        out.append(title)
    out.append(line())
    out.append(fmt_row(headers))
    out.append(line(sep_left="+", sep_mid="+", sep_right="+", pad="="))
    for r in rows:
        out.append(fmt_row(r))
    out.append(line())
    if foot:
        out.append(foot)
    return "\n".join(out)


@dataclass
class SummaryTable:
    title: str
    headers: List[str]
    rows: List[List[str]]
    foot: Optional[str] = None

    def as_text(self) -> str:
        return _draw_table(self.title, self.headers, self.rows, self.foot)

    def __str__(self) -> str:
        return self.as_text()


@dataclass
class MoranResult:
    I: float
    z: float
    n: int

    def summary(self) -> SummaryTable:
        rows = [["Moran I", _fmt(self.I)], ["z", _fmt(self.z)], ["n", str(self.n)]]
        return SummaryTable("Spatial autocorrelation", ["metric", "value"], rows)


@dataclass
class GearyResult:
    C: float
    n: int

    def summary(self) -> SummaryTable:
        rows = [["Geary C", _fmt(self.C)], ["n", str(self.n)]]
        return SummaryTable("Spatial dissimilarity", ["metric", "value"], rows)
