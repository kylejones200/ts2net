"""
Public typing protocols for ts2net.

Re-exports builder protocol for type checkers and IDE support.
"""

from __future__ import annotations

from ._builder_api import NetworkBuilder

__all__ = ["NetworkBuilder"]
