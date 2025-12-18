"""
Parity testing between R and Python implementations.

This module provides functionality to test the parity between the R and Python
implementations of various time series to network conversion algorithms.
"""
from __future__ import annotations

from .models import ParityCase, ParityReport
from .runner import run_parity_test, format_parity_report, run_r_case
from .utils import compare_graphs, nx_from_python_case, load_series, read_r_graphml

__all__ = [
    'ParityCase',
    'ParityReport',
    'run_parity_test',
    'format_parity_report',
    'run_r_case',
    'compare_graphs',
    'nx_from_python_case',
    'load_series',
    'read_r_graphml',
]
