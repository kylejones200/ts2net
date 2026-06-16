"""
Human-readable dynamic network analysis summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..graphs.dynamic import RollingGraphSequence


@dataclass
class DynamicAnalysisResult:
    """Full output of :func:`run_dynamic_analysis`."""

    sequence: RollingGraphSequence
    regime: dict[str, Any]
    anomalies: NDArray[np.float64]
    transition_anomalies: NDArray[np.float64]
    persistence: dict[tuple[int, int], float]
    churn: dict[str, NDArray[np.float64]]
    communities: dict[str, Any]
    roles: dict[int, list[str]]
    attribution: dict[str, Any] = field(default_factory=dict)
    method: str = "hvg"
    window: int = 50
    step: int = 1

    def summary(self) -> str:
        """Plain-text dynamic analysis report."""
        return format_dynamic_report(self)

    def to_markdown(self) -> str:
        """Markdown dynamic analysis report."""
        return format_dynamic_report(self, markdown=True)

    def anomalous_windows(self, threshold: float = 2.0) -> NDArray[np.int64]:
        """Window indices with anomaly score above threshold."""
        return np.where(self.anomalies >= threshold)[0].astype(np.int64)


def format_dynamic_report(
    result: DynamicAnalysisResult,
    markdown: bool = False,
) -> str:
    """
    Generate a dynamic network analysis report.

    Parameters
    ----------
    result : DynamicAnalysisResult
        Output from ``run_dynamic_analysis``.
    markdown : bool, default False
        Use markdown headings when True.

    Returns
    -------
    str
        Formatted report text.
    """
    lines: list[str] = []
    h = "## " if markdown else ""
    bullet = "- "

    n_win = len(result.sequence.stats)
    lines.append(
        f"{h}Dynamic network analysis ({result.method}, window={result.window}, "
        f"step={result.step})"
    )
    lines.append(f"{bullet}Windows analyzed: {n_win}")
    lines.append("")

    breaks = result.regime.get("break_indices", np.array([]))
    if len(breaks):
        starts = result.sequence.window_starts
        break_times = [int(starts[i]) if i < len(starts) else int(i) for i in breaks]
        lines.append(f"{h}Regime changes detected: {len(breaks)}")
        lines.append(f"{bullet}Break indices (window): {breaks.tolist()}")
        lines.append(f"{bullet}Break times (series index): {break_times}")
    else:
        lines.append(f"{h}Regime changes: none detected at current threshold")
    lines.append("")

    anom = result.anomalous_windows()
    if len(anom):
        lines.append(f"{h}Anomalous windows (score ≥ 2.0): {anom.tolist()}")
    else:
        lines.append(f"{h}Anomalous windows: none above default threshold")
    lines.append("")

    pers = result.persistence
    if pers:
        top = sorted(pers.items(), key=lambda kv: -kv[1])[:5]
        lines.append(f"{h}Most persistent edges")
        for (u, v), score in top:
            lines.append(f"{bullet}({u}, {v}): persistence={score:.2f}")
        lines.append("")

    n_comm = result.communities.get("n_communities")
    if n_comm is not None and len(n_comm):
        lines.append(
            f"{h}Communities per window: "
            f"mean={float(np.mean(n_comm)):.1f}, "
            f"range=[{int(np.min(n_comm))}, {int(np.max(n_comm))}]"
        )
        stab = result.communities.get("stability", np.array([]))
        if len(stab):
            mean_stab = float(np.mean(stab))
            lines.append(f"{bullet}Mean community stability: {mean_stab:.2f}")
        lines.append("")

    if result.attribution:
        lines.append(f"{h}Change attribution (largest metric shifts at breaks)")
        for key, val in result.attribution.items():
            lines.append(f"{bullet}{key}: {val}")
        lines.append("")

    return "\n".join(lines)
