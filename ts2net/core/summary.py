"""
Summary utilities for ts2net core module.
"""
from typing import Dict, Any, List, Union
import pandas as pd


class SummaryTable:
    """A simple summary table class for displaying results."""
    
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data or {}
    
    def add_row(self, key: str, value: Any):
        """Add a row to the summary table."""
        self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.data.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(list(self.data.items()), columns=['Metric', 'Value'])
    
    def __str__(self) -> str:
        """String representation."""
        lines = []
        for key, value in self.data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


def _fmt(value: Union[int, float, str], precision: int = 4) -> str:
    """Format a value for display."""
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)
