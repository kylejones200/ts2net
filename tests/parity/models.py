"""Data models for parity testing."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ParityCase:
    """Test case for parity testing between R and Python implementations."""

    name: str
    kind: str  # One of: HVG, NVG, RN, TN, DTW
    series: Optional[str] = None
    panel: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class ParityReport:
    """Test report for parity testing results."""

    name: str
    kind: str
    edges_jaccard: Optional[float]
    deg_l1: Optional[float]
    tri_rel_err: Optional[float]
    C_rel_err: Optional[float]
    L_rel_err: Optional[float]
    notes: str = ""
