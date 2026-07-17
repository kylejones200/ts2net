"""
Graph interpretability reports and decision packages.

Every graph should answer: what made edges exist, what changed, what is anomalous,
and what to do next.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ..causal.summary import (
    CausalAnalysisResult,
    CausalEdgeResult,
    format_causal_report,
)
from ..core import graph_summary
from ..dynamic.roles import node_roles
from ..dynamic.summary import DynamicAnalysisResult, format_dynamic_report


@dataclass
class Provenance:
    """How a graph or analysis was produced."""

    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    window: tuple[int, int] | None = None
    n_series: int | None = None
    n_points: int | None = None


@dataclass
class EdgeExplanation:
    """Why a directed or undirected edge exists."""

    source: str | int
    target: str | int
    weight: float
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    lag: int | None = None
    p_value: float | None = None
    significant: bool | None = None
    reason: str = ""

    def summary(self) -> str:
        parts = [f"{self.source} → {self.target}: weight={self.weight:.4g}"]
        if self.lag is not None:
            parts.append(f"lag={self.lag}")
        if self.p_value is not None:
            parts.append(f"p={self.p_value:.4g}")
        if self.reason:
            parts.append(self.reason)
        return ", ".join(parts)


@dataclass
class NodeRoleSummary:
    """Role and centrality context for one node."""

    node: str | int
    role: str
    degree: int
    betweenness: float | None = None
    anomaly: bool = False
    notes: str = ""


@dataclass
class GraphReport:
    """Human-readable summary of graph topology and key actors."""

    title: str
    provenance: Provenance
    topology: dict[str, Any]
    top_hubs: list[tuple[str | int, int]]
    edge_explanations: list[EdgeExplanation] = field(default_factory=list)
    node_summaries: list[NodeRoleSummary] = field(default_factory=list)
    instability_notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [self.title, ""]
        p = self.provenance
        lines.append(f"Method: {p.method}  params: {p.parameters or '{}'}")
        if p.n_points is not None:
            lines.append(f"Series length: {p.n_points}")
        lines.append("")
        lines.append("Topology")
        for key in ("n", "m", "density", "avg_degree", "avg_clustering"):
            if key in self.topology:
                val = self.topology[key]
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.4g}")
                else:
                    lines.append(f"  {key}: {val}")
        if self.top_hubs:
            lines.append("")
            lines.append("Top hubs (degree)")
            for node, deg in self.top_hubs[:8]:
                lines.append(f"  {node}: degree={deg}")
        if self.node_summaries:
            lines.append("")
            lines.append("Node roles")
            for ns in self.node_summaries[:12]:
                btw = f", betweenness={ns.betweenness:.3g}" if ns.betweenness else ""
                flag = " [anomaly]" if ns.anomaly else ""
                lines.append(f"  {ns.node}: {ns.role} (deg={ns.degree}{btw}){flag}")
        if self.edge_explanations:
            lines.append("")
            lines.append("Edge evidence (sample)")
            for ex in self.edge_explanations[:10]:
                lines.append(f"  - {ex.summary()}")
        for note in self.instability_notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        body = self.summary().replace("\n", "\n\n", 1)
        return f"## {self.title}\n\n{body}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicChangeReport:
    """Regime breaks, edge persistence, and attribution from dynamic analysis."""

    result: DynamicAnalysisResult
    title: str = "Dynamic change report"

    def summary(self) -> str:
        return cast(str, format_dynamic_report(self.result, markdown=False))

    def to_markdown(self) -> str:
        return cast(str, format_dynamic_report(self.result, markdown=True))


@dataclass
class DecisionPackage:
    """
    Evidence bundle for decision support: graph + optional causal/dynamic context.
    """

    title: str
    graph_report: GraphReport | None = None
    dynamic_report: DynamicChangeReport | None = None
    causal_result: CausalAnalysisResult | None = None
    evidence: list[str] = field(default_factory=list)
    confidence: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        sections: list[str] = [self.title, "=" * len(self.title), ""]
        if self.graph_report:
            sections.append(self.graph_report.summary())
            sections.append("")
        if self.causal_result:
            sections.append(format_causal_report(self.causal_result))
            sections.append("")
        if self.dynamic_report:
            sections.append(self.dynamic_report.summary())
            sections.append("")
        if self.evidence:
            sections.append("Evidence")
            sections.extend(f"  - {e}" for e in self.evidence)
            sections.append("")
        if self.confidence:
            sections.append("Confidence")
            sections.extend(f"  - {c}" for c in self.confidence)
            sections.append("")
        if self.assumptions:
            sections.append("Assumptions")
            sections.extend(f"  - {a}" for a in self.assumptions)
            sections.append("")
        if self.changes:
            sections.append("What changed")
            sections.extend(f"  - {c}" for c in self.changes)
            sections.append("")
        if self.next_actions:
            sections.append("Suggested next actions")
            sections.extend(f"  - {a}" for a in self.next_actions)
        return "\n".join(sections)

    def to_markdown(self) -> str:
        parts = [f"# {self.title}\n"]
        if self.graph_report:
            parts.append(self.graph_report.to_markdown())
        if self.causal_result:
            parts.append(format_causal_report(self.causal_result, markdown=True))
        if self.dynamic_report:
            parts.append(self.dynamic_report.to_markdown())
        for heading, items in (
            ("Evidence", self.evidence),
            ("Confidence", self.confidence),
            ("Assumptions", self.assumptions),
            ("What changed", self.changes),
            ("Next actions", self.next_actions),
        ):
            if items:
                parts.append(f"## {heading}\n")
                parts.extend(f"- {x}\n" for x in items)
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "graph_report": (
                self.graph_report.to_dict() if self.graph_report else None
            ),
            "dynamic_summary": (
                self.dynamic_report.summary() if self.dynamic_report else None
            ),
            "causal_summary": (
                self.causal_result.summary() if self.causal_result else None
            ),
            "evidence": self.evidence,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "changes": self.changes,
            "next_actions": self.next_actions,
        }


def _node_label(G: nx.Graph, node: Any) -> str | int:
    data = G.nodes.get(node, {})
    return cast(str | int, data.get("name", data.get("label", node)))


def _top_hubs(G: nx.Graph, k: int = 8) -> list[tuple[str | int, int]]:
    deg = dict(G.degree())
    ranked = sorted(deg.items(), key=lambda kv: -kv[1])[:k]
    return [(_node_label(G, n), int(d)) for n, d in ranked]


def _node_summaries_from_graph(G: nx.Graph) -> list[NodeRoleSummary]:
    roles = node_roles(G)
    degrees = dict(G.degree())
    try:
        btw = nx.betweenness_centrality(G)
    except Exception:
        btw = {n: 0.0 for n in G.nodes()}
    out: list[NodeRoleSummary] = []
    for node in G.nodes():
        out.append(
            NodeRoleSummary(
                node=_node_label(G, node),
                role=roles.get(node, "peripheral"),
                degree=int(degrees.get(node, 0)),
                betweenness=float(btw.get(node, 0.0)),
            )
        )
    out.sort(key=lambda s: -s.degree)
    return out


def explain_edge_from_graph(
    G: nx.Graph,
    u: Any,
    v: Any,
    method: str,
    parameters: dict[str, Any] | None = None,
) -> EdgeExplanation:
    """Build an explanation from edge attributes on ``G``."""
    data = G.get_edge_data(u, v, default={}) or {}
    weight = float(data.get("weight", 1.0))
    lag = data.get("lag")
    p_value = data.get("p_value")
    reason_parts: list[str] = []
    if data.get("method"):
        reason_parts.append(f"edge method={data['method']}")
    elif method:
        reason_parts.append(f"graph built with {method}")
    params = parameters or {}
    if "threshold" in params:
        reason_parts.append(f"threshold={params['threshold']}")
    return EdgeExplanation(
        source=_node_label(G, u),
        target=_node_label(G, v),
        weight=weight,
        method=str(data.get("method", method)),
        parameters=dict(parameters or {}),
        lag=int(lag) if lag is not None else None,
        p_value=float(p_value) if p_value is not None else None,
        significant=data.get("significant"),
        reason="; ".join(reason_parts)
        if reason_parts
        else "edge present in constructed graph",
    )


def explain_edges_from_causal(result: CausalAnalysisResult) -> list[EdgeExplanation]:
    """Convert causal workflow edges to :class:`EdgeExplanation` rows."""
    out: list[EdgeExplanation] = []
    for edge in result.edges:
        out.append(_edge_from_causal(edge, result.method, result.alpha))
    return out


def _edge_from_causal(
    edge: CausalEdgeResult,
    method: str,
    alpha: float,
) -> EdgeExplanation:
    sig_label = "significant" if edge.significant else "not significant"
    reason = f"{sig_label} at alpha={alpha}"
    if edge.adjusted:
        reason += "; confounder-adjusted"
    return EdgeExplanation(
        source=edge.source_name,
        target=edge.target_name,
        weight=edge.weight,
        method=method,
        parameters={"alpha": alpha},
        lag=edge.best_lag,
        p_value=edge.p_value,
        significant=edge.significant,
        reason=reason,
    )


def build_graph_report(
    G: nx.Graph | nx.DiGraph,
    *,
    method: str,
    parameters: dict[str, Any] | None = None,
    title: str | None = None,
    n_points: int | None = None,
    max_edges: int = 10,
) -> GraphReport:
    """
    Summarise topology, hubs, roles, and sample edge explanations.

    Parameters
    ----------
    G : networkx.Graph or DiGraph
        Built graph (optionally with edge attrs: weight, lag, p_value).
    method : str
        Builder or workflow name (e.g. ``"hvg"``, ``"granger"``).
    parameters : dict, optional
        Builder parameters for provenance.
    title : str, optional
        Report heading.
    n_points : int, optional
        Original series length.
    max_edges : int, default 10
        Number of strongest edges to explain.
    """
    params = dict(parameters or {})
    prov = Provenance(method=method, parameters=params, n_points=n_points)
    topo = graph_summary(G)
    hubs = _top_hubs(G)
    nodes = _node_summaries_from_graph(G)

    instability: list[str] = []
    density = topo.get("density", 0.0)
    if isinstance(density, float) and density > 0.5:
        instability.append(
            f"High density ({density:.3g}) — graph may be weakly informative"
        )
    if topo.get("n", 0) > 0 and topo.get("m", 0) == 0:
        instability.append("Empty edge set — check thresholds or input series")

    edge_explanations: list[EdgeExplanation] = []
    edges_with_weight: list[tuple[Any, Any, float]] = []
    for u, v, data in G.edges(data=True):
        w = abs(float(data.get("weight", 1.0)))
        edges_with_weight.append((u, v, w))
    edges_with_weight.sort(key=lambda t: -t[2])
    for u, v, _ in edges_with_weight[:max_edges]:
        edge_explanations.append(
            explain_edge_from_graph(G, u, v, method=method, parameters=params)
        )

    return GraphReport(
        title=title or f"Graph report ({method})",
        provenance=prov,
        topology=topo,
        top_hubs=hubs,
        edge_explanations=edge_explanations,
        node_summaries=nodes,
        instability_notes=instability,
    )


def build_dynamic_change_report(
    result: DynamicAnalysisResult,
    *,
    title: str | None = None,
) -> DynamicChangeReport:
    """Wrap :class:`DynamicAnalysisResult` as a change report."""
    return DynamicChangeReport(
        result=result,
        title=title or f"Dynamic change report ({result.method})",
    )


def build_decision_package(
    x: NDArray[np.float64] | None = None,
    *,
    G: nx.Graph | nx.DiGraph | None = None,
    method: str = "hvg",
    parameters: dict[str, Any] | None = None,
    window: int = 50,
    step: int = 1,
    causal: CausalAnalysisResult | None = None,
    dynamic: DynamicAnalysisResult | None = None,
    multivariate: NDArray[np.float64] | None = None,
    title: str | None = None,
) -> DecisionPackage:
    """
    Assemble evidence, confidence, assumptions, changes, and next actions.

    Provide ``G`` directly, or ``x`` (univariate) to run dynamic analysis, or
    ``multivariate`` with ``causal`` for multivariate causal packages.
    """
    from ..api import HVG, NVG
    from ..dynamic.workflow import DynamicWorkflowSpec, run_dynamic_analysis

    params = dict(parameters or {})
    assumptions = [
        f"Graph/analysis method: {method}",
        f"Parameters: {params or 'defaults'}",
    ]
    evidence: list[str] = []
    confidence: list[str] = []
    changes: list[str] = []
    next_actions: list[str] = []

    graph_report: GraphReport | None = None
    dynamic_report: DynamicChangeReport | None = None

    if G is None and x is not None and method in ("hvg", "nvg"):
        builders = {"hvg": HVG, "nvg": NVG}
        b = builders[method](**params)
        b.build(np.asarray(x, dtype=np.float64).ravel())
        G = b.as_networkx()

    if G is not None:
        graph_report = build_graph_report(
            G,
            method=method,
            parameters=params,
            n_points=len(x) if x is not None else None,
        )
        evidence.append(
            f"Graph has {graph_report.topology.get('n', 0)} nodes, "
            f"{graph_report.topology.get('m', 0)} edges"
        )
        if graph_report.top_hubs:
            top = graph_report.top_hubs[0]
            evidence.append(f"Primary hub: {top[0]} (degree={top[1]})")

    if dynamic is None and x is not None:
        spec = DynamicWorkflowSpec(
            method=(
                method
                if method in ("hvg", "nvg", "recurrence", "transition")
                else "hvg"
            ),
            window=window,
            step=step,
            builder_kwargs=params or None,
        )
        dynamic = run_dynamic_analysis(
            np.asarray(x, dtype=np.float64).ravel(),
            spec=spec,
        )

    if dynamic is not None:
        dynamic_report = build_dynamic_change_report(dynamic)
        breaks = dynamic.regime.get("break_indices", [])
        if len(breaks):
            changes.append(
                f"{len(breaks)} regime break(s) detected in rolling graph metrics"
            )
            next_actions.append("Inspect windows around detected breaks for root cause")
        anom = dynamic.anomalous_windows()
        if len(anom):
            changes.append(f"{len(anom)} anomalous graph window(s)")
            next_actions.append(
                "Review anomalous windows for sensor drift or failure precursors"
            )
        else:
            confidence.append("No anomalous windows above default threshold")

    if causal is not None:
        sig = causal.significant_edges()
        evidence.append(
            f"{len(sig)} significant causal edge(s) at alpha={causal.alpha}"
        )
        if sig:
            e = sig[0]
            evidence.append(
                f"Strongest link: {e.source_name} → {e.target_name} (p={e.p_value})"
            )
            next_actions.append("Validate top causal drivers against domain knowledge")
        else:
            next_actions.append(
                "Collect longer series or relax alpha — no significant causal links"
            )
        confidence.append(f"Causal method: {causal.method}, adjusted={causal.adjusted}")

    if multivariate is not None and causal is None:
        assumptions.append(
            "Multivariate array provided without causal result — "
            "run run_causal_analysis() for drivers"
        )

    if not next_actions:
        if graph_report and graph_report.instability_notes:
            next_actions.append("Address instability flags before downstream ML")
        else:
            next_actions.append("Monitor rolling graph metrics for regime shifts")

    return DecisionPackage(
        title=title or "ts2net decision package",
        graph_report=graph_report,
        dynamic_report=dynamic_report,
        causal_result=causal,
        evidence=evidence,
        confidence=confidence,
        assumptions=assumptions,
        changes=changes,
        next_actions=next_actions,
    )
