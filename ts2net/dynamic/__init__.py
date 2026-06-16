"""
Dynamic network analytics (horizon 0.8).

Regime detection, anomaly scoring, node role evolution, community tracking,
and an end-to-end workflow over rolling graph sequences.
"""

from .anomaly import edge_transition_anomalies, window_anomaly_scores
from .communities import community_labels, track_communities
from .regime import detect_regime_changes
from .roles import node_role_evolution, node_roles
from .summary import DynamicAnalysisResult, format_dynamic_report
from .workflow import DynamicWorkflowSpec, run_dynamic_analysis

__all__ = [
    "detect_regime_changes",
    "window_anomaly_scores",
    "edge_transition_anomalies",
    "node_roles",
    "node_role_evolution",
    "community_labels",
    "track_communities",
    "DynamicAnalysisResult",
    "format_dynamic_report",
    "DynamicWorkflowSpec",
    "run_dynamic_analysis",
]
