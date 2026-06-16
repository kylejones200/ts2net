"""
End-to-end dynamic network workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .._validation import validate_series
from ..graphs.dynamic import RollingGraphSequence
from .anomaly import edge_transition_anomalies, window_anomaly_scores
from .communities import track_communities
from .regime import detect_regime_changes
from .roles import node_role_evolution
from .summary import DynamicAnalysisResult

BuilderMethod = Literal["hvg", "nvg", "recurrence", "transition"]
RegimeMethod = Literal["zscore", "cusum"]


@dataclass
class DynamicWorkflowSpec:
    """Configuration for :func:`run_dynamic_analysis`."""

    method: BuilderMethod = "hvg"
    window: int = 50
    step: int = 1
    output: str = "stats"
    regime_metric: str = "avg_degree"
    regime_method: RegimeMethod = "zscore"
    regime_threshold: float = 2.5
    builder_kwargs: dict[str, Any] | None = None
    as_networkx: bool = True


def _stats_dict(seq: RollingGraphSequence) -> dict[str, NDArray[np.float64]]:
    if not seq.stats:
        return {}
    keys = seq.stats[0].keys()
    return {
        k: np.asarray([s.get(k, np.nan) for s in seq.stats], dtype=np.float64)
        for k in keys
    }


def _attribute_breaks(
    stats: dict[str, NDArray[np.float64]],
    break_indices: NDArray[np.int64],
) -> dict[str, float]:
    """Largest absolute metric shift at each detected break."""
    attribution: dict[str, float] = {}
    for idx in break_indices:
        i = int(idx)
        if i <= 0 or i >= len(next(iter(stats.values()))):
            continue
        best_key, best_delta = "", 0.0
        for key, arr in stats.items():
            if arr.dtype.kind not in "fi":
                continue
            delta = abs(float(arr[i]) - float(arr[i - 1]))
            if delta > best_delta:
                best_delta = delta
                best_key = key
        if best_key:
            attribution[f"window_{i}"] = f"{best_key} Δ={best_delta:.4g}"
    return attribution


def run_dynamic_analysis(
    x: NDArray[np.float64],
    spec: DynamicWorkflowSpec | None = None,
    **kwargs,
) -> DynamicAnalysisResult:
    """
    Run dynamic network analysis on a univariate time series.

    Steps:
    1. Build rolling graph sequence
    2. Detect regime changes in a chosen graph metric
    3. Score window-level and transition-level anomalies
    4. Track edge persistence, communities, and node roles
    5. Attribute metric shifts at detected breaks

    Parameters
    ----------
    x : array (n_points,)
        Input time series.
    spec : DynamicWorkflowSpec, optional
        Workflow configuration. Extra ``**kwargs`` override spec fields.
    **kwargs
        Override any ``DynamicWorkflowSpec`` field by name.

    Returns
    -------
    DynamicAnalysisResult
        Sequence, metrics, and report helpers.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> x[250:] += 3.0  # regime shift
    >>> result = run_dynamic_analysis(x, window=40, step=20)
    >>> print(result.summary())
    """
    x = validate_series(x, "run_dynamic_analysis")

    if spec is None:
        spec = DynamicWorkflowSpec()
    for key, val in kwargs.items():
        if hasattr(spec, key):
            setattr(spec, key, val)

    bkw = spec.builder_kwargs or {}
    seq = RollingGraphSequence.from_series(
        x,
        window=spec.window,
        step=spec.step,
        method=spec.method,
        output=spec.output,
        as_networkx=spec.as_networkx,
        **bkw,
    )

    stats = _stats_dict(seq)
    metric = spec.regime_metric
    if metric not in stats and stats:
        metric = next(iter(stats))
    regime_values = stats.get(metric, np.array([], dtype=np.float64))
    regime = detect_regime_changes(
        regime_values,
        method=spec.regime_method,
        threshold=spec.regime_threshold,
    )

    if stats:
        anomalies = window_anomaly_scores(stats)
    else:
        anomalies = np.array([], dtype=np.float64)
    churn = seq.churn() if seq.graphs_nx else {
        "births": np.array([], dtype=np.float64),
        "deaths": np.array([], dtype=np.float64),
        "jaccard": np.array([], dtype=np.float64),
    }
    transition_anomalies = edge_transition_anomalies(
        churn["births"], churn["deaths"], churn.get("jaccard")
    )
    persistence = seq.persistence() if seq.graphs_nx else {}
    communities = track_communities(seq.graphs_nx) if seq.graphs_nx else {
        "labels_per_window": [],
        "n_communities": np.array([], dtype=np.int64),
        "stability": np.array([], dtype=np.float64),
    }
    roles = node_role_evolution(seq.graphs_nx) if seq.graphs_nx else {}
    attribution = _attribute_breaks(stats, regime["break_indices"])  # type: ignore[arg-type]

    return DynamicAnalysisResult(
        sequence=seq,
        regime=regime,
        anomalies=anomalies,
        transition_anomalies=transition_anomalies,
        persistence=persistence,
        churn=churn,
        communities=communities,
        roles=roles,
        attribution=attribution,
        method=spec.method,
        window=spec.window,
        step=spec.step,
    )
