"""
Causal inference and network-based causality analysis.

This module provides methods for inferring causal relationships between time series
using information-theoretic and statistical approaches:
- Transfer entropy: Information-theoretic causality measure
- Granger causality: Statistical causality tests (linear and nonlinear)
- Causal networks: Directed networks based on causal relationships
- Causal metrics: Path-based causality measures and directionality indices
- Time-lagged analysis: Causal structure across multiple lags
- Workflow: End-to-end lag search, confidence, confounders, and summaries
"""

from .transfer_entropy import (
    transfer_entropy,
    transfer_entropy_network,
    conditional_transfer_entropy,
)
from .granger import granger_causality, granger_causality_network
from .metrics import causal_strength, directionality_index, causal_network_metrics
from .time_lagged import time_lagged_causality_network
from .lag_search import search_granger_lag, search_te_lag
from .confidence import te_permutation_test, te_bootstrap_ci
from .confounders import partial_granger_causality, conditional_te_network
from .summary import CausalEdgeResult, CausalAnalysisResult, format_causal_report
from .workflow import CausalWorkflowSpec, run_causal_analysis

__all__ = [
    "transfer_entropy",
    "transfer_entropy_network",
    "conditional_transfer_entropy",
    "granger_causality",
    "granger_causality_network",
    "causal_strength",
    "directionality_index",
    "causal_network_metrics",
    "time_lagged_causality_network",
    "search_granger_lag",
    "search_te_lag",
    "te_permutation_test",
    "te_bootstrap_ci",
    "partial_granger_causality",
    "conditional_te_network",
    "CausalEdgeResult",
    "CausalAnalysisResult",
    "format_causal_report",
    "CausalWorkflowSpec",
    "run_causal_analysis",
]
