"""
Expanded graph construction families (v0.4 core graph expansion).

High-level builders for correlation, similarity, dynamic, multiplex, and
extended recurrence/transition/visibility methods.
"""

from .correlation import (
    correlation_matrix,
    correlation_network,
    partial_correlation_matrix,
    partial_correlation_network,
    rolling_correlation_matrix,
    rolling_correlation_network,
)
from .dynamic import (
    RollingGraphSequence,
    edge_birth_death,
    edge_persistence,
    graph_churn,
)
from .events import event_sequence_network, event_sync_network
from .multiplex import MultiplexGraph, multiplex_graph
from .recurrence import (
    adaptive_recurrence_network,
    cross_recurrence_network,
    recurrence_matrix,
    recurrence_quantification,
)
from .similarity import similarity_matrix, similarity_network
from .transition import entropy_max_symbolize, sax_symbolize, sax_transition_network
from .visibility import multiplex_visibility_graph

__all__ = [
    "correlation_matrix",
    "correlation_network",
    "partial_correlation_matrix",
    "partial_correlation_network",
    "rolling_correlation_matrix",
    "rolling_correlation_network",
    "similarity_matrix",
    "similarity_network",
    "RollingGraphSequence",
    "edge_persistence",
    "graph_churn",
    "edge_birth_death",
    "MultiplexGraph",
    "multiplex_graph",
    "multiplex_visibility_graph",
    "adaptive_recurrence_network",
    "cross_recurrence_network",
    "recurrence_matrix",
    "recurrence_quantification",
    "event_sequence_network",
    "event_sync_network",
    "sax_symbolize",
    "entropy_max_symbolize",
    "sax_transition_network",
]
