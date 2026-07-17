"""
Public API stability tiers for ts2net v1.0.

Import from this module to introspect which symbols are stable, experimental,
or deprecated. See ``docs/API_STABILITY.md`` for policy details.
"""

from __future__ import annotations

STABLE: frozenset[str] = frozenset(
    {
        "Graph",
        "HVG",
        "NVG",
        "RecurrenceNetwork",
        "TransitionNetwork",
        "build_network",
        "build_windows",
        "graph_summary",
        "NetworkBuilder",
        "NotBuiltError",
        "ValidationError",
        "Ts2NetError",
        "run_causal_analysis",
        "CausalAnalysisResult",
        "CausalWorkflowSpec",
        "run_dynamic_analysis",
        "DynamicAnalysisResult",
        "DynamicWorkflowSpec",
        "GraphReport",
        "EdgeExplanation",
        "NodeRoleSummary",
        "DynamicChangeReport",
        "DecisionPackage",
        "Provenance",
        "build_graph_report",
        "build_dynamic_change_report",
        "build_decision_package",
        "explain_edge_from_graph",
        "explain_edges_from_causal",
    }
)

EXPERIMENTAL: frozenset[str] = frozenset(
    {
        "PipelineConfig",
        "create_graph_builder",
        "build_graph_from_config",
        "fit_sindy",
        "SINDySpec",
        "SINDyResult",
        "NeuralNetworkInference",
        "temporal_cnn_embeddings",
        "compare_feature_sets",
        "MultiplexGraph",
        "pc_algorithm",
        "fci_algorithm",
    }
)

DEPRECATED: frozenset[str] = frozenset(
    {
        # Reserved for v1.0 removals; empty until deprecation notices ship.
    }
)


def api_tier(name: str) -> str:
    """Return ``stable``, ``experimental``, ``deprecated``, or ``internal``."""
    if name in DEPRECATED:
        return "deprecated"
    if name in STABLE:
        return "stable"
    if name in EXPERIMENTAL:
        return "experimental"
    return "internal"
