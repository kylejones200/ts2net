"""
Input validation and preprocessing for parity testing.

Enforces consistent rules across all builders.
"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_timeseries(
    x: np.ndarray,
    min_length: int = 3,
    strict_validation: bool = False,
    allow_constant: bool = False,
    name: str = "x"
) -> np.ndarray:
    """
    Apply consistent preprocessing rules to time series input.
    
    Default behavior (strict_validation=False): Drop NaN/Inf with logging (R-like)
    Strict mode (strict_validation=True): Raise error on NaN/Inf
    
    Args:
        x: Input array
        min_length: Minimum required length
        strict_validation: If True, raise error on NaN/Inf; if False, drop with warning
        allow_constant: If False, raise error on constant series
        name: Variable name for error messages
        
    Returns:
        Validated and preprocessed array (always float64, 1D)
        
    Raises:
        ValidationError: If validation fails
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Rule 1: Must be 1D after squeeze
    x = x.squeeze()
    if x.ndim != 1:
        raise ValidationError(f"{name} must be 1D array, got shape {x.shape}")
    
    # Rule 2: Check initial length
    if len(x) < min_length:
        raise ValidationError(
            f"{name} too short: length {len(x)} < minimum {min_length}"
        )
    
    # Rule 3: Handle NaN
    n_nan = np.sum(np.isnan(x))
    if n_nan > 0:
        if strict_validation:
            raise ValidationError(f"{name} contains {n_nan} NaN values")
        else:
            logger.warning(f"{name}: Dropping {n_nan} NaN values")
            x = x[~np.isnan(x)]
            if len(x) < min_length:
                raise ValidationError(
                    f"{name} too short after dropping NaN: {len(x)} < {min_length}"
                )
    
    # Rule 4: Handle Inf
    n_inf = np.sum(np.isinf(x))
    if n_inf > 0:
        if strict_validation:
            raise ValidationError(f"{name} contains {n_inf} Inf values")
        else:
            logger.warning(f"{name}: Dropping {n_inf} Inf values")
            x = x[~np.isinf(x)]
            if len(x) < min_length:
                raise ValidationError(
                    f"{name} too short after dropping Inf: {len(x)} < {min_length}"
                )
    
    # Rule 5: Check for constant series
    if not allow_constant:
        if np.allclose(x, x[0], rtol=1e-10, atol=1e-10):
            raise ValidationError(f"{name} is constant (all values ≈ {x[0]:.6f})")
    
    return x


def validate_embedding_params(m: int, tau: int, n: int) -> None:
    """
    Validate embedding parameters.
    
    Args:
        m: Embedding dimension
        tau: Time delay
        n: Time series length
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if m < 1:
        raise ValidationError(f"Embedding dimension m must be ≥ 1, got {m}")
    
    if tau < 1:
        raise ValidationError(f"Time delay tau must be ≥ 1, got {tau}")
    
    # Check if embedding is possible
    required_length = (m - 1) * tau + 1
    if n < required_length:
        raise ValidationError(
            f"Series too short for embedding: n={n} < required={(m-1)*tau+1} "
            f"for m={m}, tau={tau}"
        )


def validate_transition_params(order: int, delay: int, n: int) -> None:
    """
    Validate transition network parameters.
    
    Args:
        order: Ordinal pattern length
        delay: Time delay
        n: Time series length
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if order < 2:
        raise ValidationError(f"Order must be ≥ 2, got {order}")
    
    if delay < 1:
        raise ValidationError(f"Delay must be ≥ 1, got {delay}")
    
    # Check if pattern extraction is possible
    required_length = (order - 1) * delay + 1
    if n < required_length:
        raise ValidationError(
            f"Series too short for ordinal patterns: n={n} < required={(order-1)*delay+1} "
            f"for order={order}, delay={delay}"
        )


def normalize_metric_name(metric: str) -> str:
    """
    Normalize metric names for R/Python compatibility.
    
    R ts2net uses "maximum" for Chebyshev distance.
    Python uses "chebyshev".
    
    Args:
        metric: Metric name (euclidean|manhattan|maximum|chebyshev)
        
    Returns:
        Normalized metric name for Python
    """
    _METRIC_ALIASES = {
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "maximum": "chebyshev",  # R → Python
        "chebyshev": "chebyshev",
    }
    
    metric_lower = metric.lower()
    if metric_lower not in _METRIC_ALIASES:
        raise ValidationError(
            f"Unknown metric: {metric}. "
            f"Must be one of: {list(_METRIC_ALIASES.keys())}"
        )
    
    return _METRIC_ALIASES[metric_lower]


def check_tie_handling(x: np.ndarray, method: str = "ordinal") -> dict:
    """
    Analyze tie structure in time series.
    
    Returns statistics about repeated values that may affect
    ordinal pattern extraction or visibility graphs.
    
    Args:
        x: Time series
        method: Analysis method
        
    Returns:
        Dictionary with tie statistics
    """
    unique_vals = np.unique(x)
    n_unique = len(unique_vals)
    n_total = len(x)
    
    # Count ties
    tie_counts = {}
    for val in unique_vals:
        count = np.sum(x == val)
        if count > 1:
            tie_counts[val] = count
    
    n_tied_values = len(tie_counts)
    n_tied_points = sum(tie_counts.values())
    
    return {
        "n_unique": n_unique,
        "n_total": n_total,
        "uniqueness_ratio": n_unique / n_total,
        "n_tied_values": n_tied_values,
        "n_tied_points": n_tied_points,
        "tie_ratio": n_tied_points / n_total,
        "max_tie_count": max(tie_counts.values()) if tie_counts else 1,
        "has_ties": n_tied_values > 0,
    }


def preprocess_for_method(
    x: np.ndarray,
    method: str,
    params: Optional[dict] = None
) -> np.ndarray:
    """
    Apply method-specific preprocessing.
    
    Args:
        x: Input time series
        method: Method name (HVG|NVG|RN|TN)
        params: Method parameters
        
    Returns:
        Preprocessed time series
    """
    params = params or {}
    
    # All methods: basic validation
    x = validate_timeseries(
        x,
        min_length=3,
        allow_nan=False,
        allow_inf=False,
        allow_constant=False
    )
    
    # Method-specific validation
    if method in ("HVG", "NVG"):
        # Visibility graphs work on any non-constant series
        pass
    
    elif method == "RN":
        # Recurrence networks need embedding validation
        m = params.get("m", 2)
        tau = params.get("tau", 1)
        validate_embedding_params(m, tau, len(x))
    
    elif method == "TN":
        # Transition networks need pattern extraction validation
        order = params.get("order", 3)
        delay = params.get("delay", 1)
        validate_transition_params(order, delay, len(x))
    
    return x

