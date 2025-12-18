"""
Event detection and synchronization for time series.

This module provides functions for detecting events in time series data
and computing event synchronization measures.
"""

from .core import (
    events_from_ts,
    tssim_event_sync,
    random_ets,
    event_sync_full,
    EventSyncResult,
)

__all__ = [
    "events_from_ts",
    "tssim_event_sync",
    "random_ets",
    "event_sync_full",
    "EventSyncResult",
]
