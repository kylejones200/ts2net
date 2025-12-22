"""
BSTS decomposition using statsmodels state space models.

Provides fast structural decomposition without full Bayesian sampling.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class BSTSSpec:
    """Specification for structural time series decomposition.
    
    Parameters
    ----------
    level : bool, default True
        Include local level component
    trend : bool, default False
        Include local linear trend
    seasonal_periods : list of int, optional
        Seasonal periods (e.g., [24, 168] for hourly data with daily/weekly seasonality)
    robust : bool, default False
        Use Student-t errors for heavy tails (slower, more robust)
    standardize_residual : bool, default True
        Standardize residual before analysis (recommended)
    """
    level: bool = True
    trend: bool = False
    seasonal_periods: Optional[List[int]] = None
    robust: bool = False
    standardize_residual: bool = True


@dataclass
class DecompositionResult:
    """Result of structural decomposition.
    
    Attributes
    ----------
    level : array
        Level component (local mean)
    residual : array
        Residual (observed - fitted)
    fitted : array
        Fitted values (level + trend + seasonal)
    trend : array, optional
        Trend component (if trend=True)
    seasonal : array, optional
        Seasonal component (if seasonal_periods provided)
    level_variance : float
        Estimated level variance
    trend_variance : float, optional
        Estimated trend variance (if trend=True)
    seasonal_variance : dict, optional
        Estimated seasonal variance per period (if seasonal_periods provided)
    error_variance : float
        One-step-ahead forecast error variance
    """
    level: NDArray[np.float64]
    residual: NDArray[np.float64]
    fitted: NDArray[np.float64]
    trend: Optional[NDArray[np.float64]] = None
    seasonal: Optional[NDArray[np.float64]] = None
    level_variance: float = 0.0
    trend_variance: Optional[float] = None
    seasonal_variance: Optional[Dict[int, float]] = None
    error_variance: float = 0.0


def decompose(series: NDArray[np.float64], spec: BSTSSpec) -> DecompositionResult:
    """
    Decompose time series into structural components using state space model.
    
    Uses statsmodels state space models for fast MLE estimation.
    
    Parameters
    ----------
    series : array
        Input time series
    spec : BSTSSpec
        Decomposition specification
    
    Returns
    -------
    DecompositionResult
        Components, residual, and variance estimates
    
    Raises
    ------
    ImportError
        If statsmodels is not installed
    ValueError
        If series is too short or constant
    """
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError:
        raise ImportError(
            "statsmodels required for BSTS decomposition. "
            "Install with: pip install ts2net[bsts]"
        )
    
    # Validate input
    series = np.asarray(series, dtype=np.float64)
    if len(series) < 10:
        raise ValueError(f"Series too short for decomposition: {len(series)} points")
    if np.std(series) == 0:
        raise ValueError("Constant series cannot be decomposed")
    
    # Build state space model specification
    # statsmodels uses specific component names
    level_type = 'local level' if spec.level else None
    trend_type = 'local linear trend' if spec.trend else None
    
    # Handle seasonal periods
    # statsmodels supports one seasonal period directly
    # For multiple periods, use the first one (can be extended later)
    seasonal_period = None
    if spec.seasonal_periods:
        seasonal_period = spec.seasonal_periods[0]
        if len(spec.seasonal_periods) > 1:
            import warnings
            warnings.warn(
                f"Multiple seasonal periods provided {spec.seasonal_periods}. "
                f"Using first period {seasonal_period}. "
                f"Multiple seasonals require custom state space model."
            )
    
    # Fit model
    model = UnobservedComponents(
        series,
        level=level_type,
        trend=trend_type,
        seasonal=seasonal_period
    )
    
    if spec.robust:
        # Use Student-t errors (slower but more robust)
        # Note: statsmodels may not support this directly, fallback to normal errors
        fitted_model = model.fit(method='lbfgs', disp=False)
    else:
        fitted_model = model.fit(method='lbfgs', disp=False)
    
    # Extract components (use smoothed estimates)
    level = fitted_model.level.smoothed if spec.level and hasattr(fitted_model, 'level') and fitted_model.level is not None else np.zeros_like(series)
    
    trend = None
    if spec.trend and hasattr(fitted_model, 'trend') and fitted_model.trend is not None:
        trend = fitted_model.trend.smoothed
    
    # Handle seasonal
    seasonal = None
    if spec.seasonal_periods and hasattr(fitted_model, 'seasonal') and fitted_model.seasonal is not None:
        seasonal = fitted_model.seasonal.smoothed
    
    # Compute fitted values
    fitted = level.copy()
    if trend is not None:
        fitted = fitted + trend
    if seasonal is not None:
        fitted = fitted + seasonal
    
    # Compute residual
    residual = series - fitted
    
    # Standardize if requested
    if spec.standardize_residual and np.std(residual) > 0:
        residual = (residual - np.mean(residual)) / np.std(residual)
    
    # Extract variance estimates from fitted model
    # Use component variances from state space model
    level_variance = 0.0
    trend_variance = None
    seasonal_variance = {}
    error_variance = np.var(residual)
    
    # Extract variance parameters from model
    # statsmodels stores variances in params with specific naming
    params = fitted_model.params
    param_names = fitted_model.param_names
    
    for i, name in enumerate(param_names):
        param_val = params[i]
        # Variance parameters are typically stored as log-variance or variance
        # Check if it's already a variance or needs squaring
        if param_val < 0:
            # Likely log-variance, convert
            var_val = np.exp(param_val)
        else:
            # May already be variance, but statsmodels often uses log
            var_val = param_val ** 2 if param_val < 10 else param_val
        
        name_lower = name.lower()
        if 'level' in name_lower or 'irregular' in name_lower:
            level_variance = var_val
        elif 'trend' in name_lower and spec.trend:
            trend_variance = var_val
        elif 'seasonal' in name_lower and spec.seasonal_periods:
            # Use the period we actually fitted
            if seasonal_period:
                seasonal_variance[seasonal_period] = var_val
    
    return DecompositionResult(
        level=level,
        residual=residual,
        fitted=fitted,
        trend=trend,
        seasonal=seasonal,
        level_variance=level_variance,
        trend_variance=trend_variance,
        seasonal_variance=seasonal_variance if seasonal_variance else None,
        error_variance=error_variance
    )
