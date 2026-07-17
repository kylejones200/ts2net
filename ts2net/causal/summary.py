"""
Human-readable causal analysis summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .metrics import causal_network_metrics


@dataclass
class CausalEdgeResult:
    """Single directed causal edge with evidence."""

    source: int
    target: int
    source_name: str
    target_name: str
    weight: float
    p_value: Optional[float] = None
    best_lag: Optional[int] = None
    significant: bool = True
    adjusted: bool = False
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


@dataclass
class CausalAnalysisResult:
    """Full output of :func:`run_causal_analysis`."""

    graph: nx.DiGraph
    edges: List[CausalEdgeResult]
    matrix: NDArray[np.float64]
    metrics: Dict[str, object]
    method: str
    lag_by_pair: Dict[Tuple[int, int], int] = field(default_factory=dict)
    alpha: float = 0.05
    adjusted: bool = False
    series_names: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Plain-text causal report."""
        return format_causal_report(self)

    def to_markdown(self) -> str:
        """Markdown causal report."""
        return format_causal_report(self, markdown=True)

    def significant_edges(self) -> List[CausalEdgeResult]:
        """Edges passing the significance threshold."""
        return [e for e in self.edges if e.significant]


def format_causal_report(
    result: CausalAnalysisResult,
    markdown: bool = False,
) -> str:
    """
    Generate a causal explanation report.

    Parameters
    ----------
    result : CausalAnalysisResult
        Output from ``run_causal_analysis``.
    markdown : bool, default False
        Use markdown headings when True.

    Returns
    -------
    str
        Formatted report text.
    """
    lines: List[str] = []
    h = "## " if markdown else ""
    bullet = "- "

    lines.append(f"{h}Causal network analysis ({result.method})")
    lines.append(
        f"Nodes: {result.metrics.get('n_nodes', result.graph.number_of_nodes())}, "
        f"edges: {len(result.significant_edges())} significant "
        f"(alpha={result.alpha})"
    )
    if result.adjusted:
        lines.append("Confounder adjustment: enabled")
    lines.append("")

    raw_emitters = result.metrics.get("top_emitters")
    raw_receivers = result.metrics.get("top_receivers")
    emitters: list[int] = (
        [int(i) for i in raw_emitters] if isinstance(raw_emitters, list) else []
    )
    receivers: list[int] = (
        [int(i) for i in raw_receivers] if isinstance(raw_receivers, list) else []
    )
    if emitters or receivers:
        lines.append(f"{h}Network roles")
        if emitters:
            names = [_node_name(result, i) for i in emitters]
            lines.append(f"{bullet}Top causal emitters: {', '.join(names)}")
        if receivers:
            names = [_node_name(result, i) for i in receivers]
            lines.append(f"{bullet}Top causal receivers: {', '.join(names)}")
        lines.append("")

    sig = result.significant_edges()
    if sig:
        lines.append(f"{h}Significant causal links")
        for edge in sorted(sig, key=lambda e: e.p_value if e.p_value is not None else 0.0):
            lag_txt = f", lag={edge.best_lag}" if edge.best_lag is not None else ""
            p_txt = f", p={edge.p_value:.4f}" if edge.p_value is not None else ""
            adj_txt = " (confounder-adjusted)" if edge.adjusted else ""
            lines.append(
                f"{bullet}{edge.source_name} → {edge.target_name}: "
                f"strength={edge.weight:.4f}{p_txt}{lag_txt}{adj_txt}"
            )
    else:
        lines.append("No significant causal edges detected at the chosen threshold.")

    lines.append("")
    return "\n".join(lines)


def _node_name(result: CausalAnalysisResult, idx: int) -> str:
    if result.series_names and idx < len(result.series_names):
        return result.series_names[idx]
    data = result.graph.nodes.get(idx, {})
    return str(data.get("name", f"Series_{idx}"))
