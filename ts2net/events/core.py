"""
Core event detection and synchronization functions.

This module provides functions for detecting events in time series data
and computing event synchronization measures.
"""

from typing import Optional, Tuple, List, Union
import numpy as np
import networkx as nx
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class EventSyncResult:
    """Container for event synchronization results."""

    q12: float
    q21: float
    q: float
    c12: int
    c21: int
    ties: int
    delays: Optional[np.ndarray] = None


def events_from_ts(
    x: np.ndarray,
    method: str = "threshold",
    thresh: Optional[float] = None,
    min_separation: int = 1,
) -> np.ndarray:
    """
    Detect events in a time series.

    Args:
        x: Input time series
        method: Detection method ('threshold' or 'peaks')
        thresh: Threshold for event detection (required for 'threshold' method)
        min_separation: Minimum samples between consecutive events

    Returns:
        Array of event indices
    """
    if method == "threshold":
        if thresh is None:
            thresh = np.median(x) + np.std(x)
        events = np.where(x > thresh)[0]
    elif method == "peaks":
        if thresh is None:
            thresh = 0.5  # Default relative height for peak detection
        events, _ = find_peaks(x, height=thresh, distance=min_separation)
    else:
        raise ValueError(f"Unknown event detection method: {method}")

    # Ensure minimum separation
    if len(events) > 1 and min_separation > 1:
        mask = np.ones(len(events), dtype=bool)
        for i in range(1, len(events)):
            if events[i] - events[i - 1] < min_separation:
                mask[i] = False
        events = events[mask]

    return events


def tssim_event_sync(
    e1: np.ndarray, e2: np.ndarray, adaptive: bool = True, tau_fixed: int = 1
) -> Tuple[float, float, float]:
    """
    Compute event synchronization between two event sequences.

    Args:
        e1, e2: Arrays of event times
        adaptive: If True, use adaptive time window
        tau_fixed: Fixed time window (used if adaptive=False)

    Returns:
        Tuple of (q12, q21, q) where:
        - q12: Synchronization from e1 to e2
        - q21: Synchronization from e2 to e1
        - q: Total synchronization
    """
    if len(e1) == 0 or len(e2) == 0:
        return 0.0, 0.0, 0.0

    c12 = 0  # e1 -> e2
    c21 = 0  # e2 -> e1
    ties = 0

    i = j = 0
    n1, n2 = len(e1), len(e2)

    while i < n1 and j < n2:
        dt = e1[i] - e2[j]

        if adaptive:
            # Adaptive time window
            tau = min(_neighbor_gap(e1, i, n1), _neighbor_gap(e2, j, n2)) / 2
        else:
            tau = tau_fixed

        if abs(dt) <= tau:
            if dt < 0:
                c12 += 1.0
            elif dt > 0:
                c21 += 1.0
            else:
                c12 += 0.5
                c21 += 0.5
                ties += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    # Normalize by number of events
    q12 = c12 / n2 if n2 > 0 else 0.0
    q21 = c21 / n1 if n1 > 0 else 0.0
    q = (c12 + c21) / (n1 + n2) if (n1 + n2) > 0 else 0.0

    return q12, q21, q


def _neighbor_gap(events: np.ndarray, idx: int, n: int) -> float:
    """
    Compute the time gap to the nearest neighbor.

    Args:
        events: Array of event times
        idx: Current event index
        n: Length of events array

    Returns:
        Time gap to nearest neighbor
    """
    if n <= 1:
        return 1.0  # Default gap if only one event

    if idx == 0:
        return events[1] - events[0]
    elif idx == n - 1:
        return events[-1] - events[-2]
    else:
        return min(events[idx] - events[idx - 1], events[idx + 1] - events[idx])


def random_ets(
    n_events: int, T: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate random event times.

    Args:
        n_events: Number of events to generate
        T: Time range [0, T)
        rng: Optional random number generator

    Returns:
        Sorted array of event times
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_events == 0:
        return np.array([], dtype=int)

    # Generate random event times and sort them
    events = np.sort(rng.integers(0, T, size=n_events))

    # Ensure unique times
    if len(np.unique(events)) < len(events):
        # If duplicates exist, generate more events and take first n_events unique ones
        while True:
            extra = rng.integers(0, T, size=n_events)
            events = np.unique(np.concatenate([events, extra]))
            if len(events) >= n_events:
                events = np.sort(events[:n_events])
                break

    return events


def event_sync_full_summary(
    e1: np.ndarray,
    e2: np.ndarray,
    adaptive: bool = True,
    tau_max: Optional[float] = None,
    normalize: str = "sqrt",
) -> EventSyncResult:
    """
    Wrapper for event_sync_full that returns an EventSyncResult object.
    
    Args:
        e1, e2: Arrays of event times
        adaptive: If True, use adaptive time window
        tau_max: Maximum time window (used if adaptive=False)
        normalize: Normalization method ('sqrt', 'min', or 'none')
        
    Returns:
        EventSyncResult object with synchronization measures
    """
    res = event_sync_full(
        e1,
        e2,
        adaptive=adaptive,
        tau_max=tau_max,
        normalize=normalize,
        return_details=False,
    )
    return EventSyncResult(
        n1=int(len(e1)),
        n2=int(len(e2)),
        c12=float(res["c12"]),
        c21=float(res["c21"]),
        ties=float(res["ties"]),
        q12=float(res["q12"]),
        q21=float(res["q21"]),
        Q=float(res["Q"]),
    )


def event_sync_full(
    e1: np.ndarray,
    e2: np.ndarray,
    adaptive: bool = True,
    tau_max: Optional[float] = None,
    normalize: str = "sqrt",
    return_details: bool = False,
) -> Union[EventSyncResult, dict]:
    """
    Compute event synchronization between two event sequences with detailed results.

    Args:
        e1, e2: Arrays of event times
        adaptive: If True, use adaptive time window
        tau_max: Maximum time window (used if adaptive=False)
        normalize: Normalization method ('sqrt', 'min', or 'none')
        return_details: If True, return detailed results

    Returns:
        EventSyncResult or dict with synchronization measures
    """
    if len(e1) == 0 or len(e2) == 0:
        result = EventSyncResult(
            q12=0.0,
            q21=0.0,
            q=0.0,
            c12=0,
            c21=0,
            ties=0,
            delays=np.array([]) if return_details else None,
        )
        return result if return_details else result._asdict()

    c12 = 0.0  # e1 -> e2
    c21 = 0.0  # e2 -> e1
    ties = 0.0
    delays = []

    i = j = 0
    n1, n2 = len(e1), len(e2)

    while i < n1 and j < n2:
        dt = e1[i] - e2[j]

        if adaptive:
            # Adaptive time window
            tau1 = _neighbor_gap(e1, i, n1) / 2
            tau2 = _neighbor_gap(e2, j, n2) / 2
            tau = min(tau1, tau2)
        else:
            tau = tau_max or 1.0

        if abs(dt) <= tau:
            if dt < 0:
                c12 += 1.0
            elif dt > 0:
                c21 += 1.0
            else:
                c12 += 0.5
                c21 += 0.5
                ties += 1.0

            if return_details:
                delays.append(float(dt))

            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    # Normalize
    if normalize == "sqrt":
        norm = (n1 * n2) ** 0.5
    elif normalize == "min":
        norm = float(min(n1, n2))
    elif normalize == "none":
        norm = 1.0
    else:
        raise ValueError("normalize must be 'sqrt', 'min', or 'none'.")

    q12 = c12 / norm if norm > 0 else 0.0
    q21 = c21 / norm if norm > 0 else 0.0
    q = (c12 + c21) / norm if norm > 0 else 0.0

    result = EventSyncResult(
        q12=q12,
        q21=q21,
        q=q,
        c12=int(c12),
        c21=int(c21),
        ties=int(ties),
        delays=np.array(delays, float) if return_details else None,
    )

    return result if return_details else result._asdict()
