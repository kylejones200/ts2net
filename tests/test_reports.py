"""Tests for graph reports and decision packages."""

from __future__ import annotations

import numpy as np
import pytest

from ts2net import HVG
from ts2net.causal import CausalAnalysisResult, CausalEdgeResult
from ts2net.dynamic import DynamicWorkflowSpec, run_dynamic_analysis
from ts2net.reports import (
    build_decision_package,
    build_graph_report,
    explain_edges_from_causal,
)


class TestGraphReport:
    def test_build_graph_report_hvg(self):
        rng = np.random.default_rng(0)
        x = np.cumsum(rng.normal(0, 1, 120))
        b = HVG()
        b.build(x)
        G = b.as_networkx()
        report = build_graph_report(G, method="hvg", n_points=len(x))
        text = report.summary()
        assert "hvg" in text.lower() or "Graph report" in text
        assert report.topology["n"] == len(x)
        assert len(report.top_hubs) >= 1
        assert len(report.node_summaries) == len(x)

    def test_explain_edges_from_causal(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edge(0, 1, weight=0.8)
        edge = CausalEdgeResult(
            source=0,
            target=1,
            source_name="A",
            target_name="B",
            weight=0.8,
            p_value=0.01,
            best_lag=2,
            significant=True,
        )
        result = CausalAnalysisResult(
            graph=G,
            edges=[edge],
            matrix=np.zeros((2, 2)),
            metrics={"n_nodes": 2},
            method="granger",
            alpha=0.05,
            series_names=["A", "B"],
        )
        explained = explain_edges_from_causal(result)
        assert len(explained) == 1
        assert explained[0].lag == 2
        assert "significant" in explained[0].reason


class TestDecisionPackage:
    def test_univariate_package(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.sin(t) + 0.1 * np.random.default_rng(1).normal(size=len(t))
        pkg = build_decision_package(
            x,
            method="hvg",
            window=40,
            step=5,
            title="Sensor check",
        )
        assert pkg.graph_report is not None
        assert pkg.dynamic_report is not None
        md = pkg.to_markdown()
        assert "Sensor check" in md
        assert pkg.next_actions

    def test_dynamic_only(self):
        x = np.linspace(0, 1, 100) ** 2
        dynamic = run_dynamic_analysis(
            x,
            spec=DynamicWorkflowSpec(method="hvg", window=25, step=2),
        )
        pkg = build_decision_package(x=None, G=None, dynamic=dynamic, method="hvg")
        assert "Dynamic" in pkg.summary() or "dynamic" in pkg.summary().lower()
