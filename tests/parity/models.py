"""
Data models for parity testing between R and Python implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ParityCase:
    """Test case for parity testing between R and Python implementations."""
    name: str
    kind: str  # One of: HVG, NVG, RN, TN, DTW
    series: str | None = None
    panel: str | None = None
    params: Dict[str, Any] = None


@dataclass
class ParityReport:
    """Test report for parity testing results."""
    name: str
    kind: str
    edges_jaccard: float | None
    deg_l1: float | None
    tri_rel_err: float | None
    C_rel_err: float | None
    L_rel_err: float | None
    notes: str = ""
