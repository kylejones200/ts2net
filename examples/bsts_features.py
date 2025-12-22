"""
Example: BSTS Decomposition and Residual Topology Analysis

Demonstrates structural time series decomposition and residual network analysis:
- Decompose series into level, trend, and seasonal components
- Analyze residual with visibility and transition graphs
- Compare meters/wells without seasonal confounds
- Flag series where structural model fails (high residual complexity)
"""

import sys
import os
# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from pathlib import Path

try:
    from ts2net.bsts import features, BSTSSpec
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Install statsmodels: pip install ts2net[bsts]")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def example_smart_meter():
    """Smart meter example with daily/weekly seasonality."""
    logger.info("Smart Meter Example: Daily/Weekly Seasonality")
    
    # Generate synthetic hourly smart meter data with seasonality
    np.random.seed(42)
    n_hours = 24 * 30  # 30 days of hourly data
    
    # Create time index
    t = np.arange(n_hours)
    
    # Level (slowly varying baseline)
    level = 50 + 0.1 * t / 24  # Gradual increase
    
    # Daily seasonality (24-hour cycle)
    daily = 10 * np.sin(2 * np.pi * t / 24)
    
    # Weekly seasonality (168-hour cycle)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    
    # Residual (irregular dynamics)
    residual = np.random.randn(n_hours) * 2
    
    # Observed series
    x = level + daily + weekly + residual
    
    logger.info(f"Generated {len(x)} hourly points")
    
    # Decompose with BSTS
    spec = BSTSSpec(
        level=True,
        trend=False,
        seasonal_periods=[24, 168],  # Daily and weekly
        standardize_residual=True
    )
    
    logger.info("Decomposing with BSTS...")
    result = features(x, methods=['hvg', 'transition'], bsts=spec)
    
    # Display results
    logger.info("\nRaw Statistics:")
    for key, value in result.raw_stats.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    logger.info("\nStructural Statistics:")
    for key, value in result.structural_stats.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    logger.info("\nResidual Network Statistics:")
    for method, stats in result.residual_network_stats.items():
        logger.info(f"  {method}:")
        for key, value in stats.items():
            logger.info(f"    {key}: {value:.4f}" if isinstance(value, float) else f"    {key}: {value}")
    
    # Interpretation
    seasonal_strength = result.structural_stats.get('seasonal_strength', 0)
    residual_complexity = result.residual_network_stats.get('hvg', {}).get('avg_degree', 0)
    
    logger.info(f"\nInterpretation:")
    logger.info(f"  Seasonal strength: {seasonal_strength:.2%}")
    logger.info(f"  Residual complexity (HVG avg degree): {residual_complexity:.2f}")
    if residual_complexity > 3.5:
        logger.info("  High residual complexity - structural model may be missing dynamics")
    else:
        logger.info("  Low residual complexity - structural model captures most dynamics")


def example_well_production():
    """Well production example with decline trend."""
    logger.info("\nWell Production Example: Decline Trend")
    
    # Generate synthetic monthly production data with decline
    np.random.seed(42)
    n_months = 60  # 5 years
    
    # Arps-like decline
    t = np.arange(n_months)
    qi = 1000  # Initial rate
    Di = 0.1  # Decline rate
    b = 1.2  # Decline exponent
    
    # Decline curve
    decline = qi / (1 + b * Di * t) ** (1 / b)
    
    # Add noise and occasional regime shifts
    noise = np.random.randn(n_months) * 50
    # Add a regime shift at month 30
    regime_shift = np.where(t >= 30, 200, 0)
    
    # Observed production
    x = decline + noise + regime_shift
    
    logger.info(f"Generated {len(x)} monthly points")
    
    # Decompose with BSTS (level + trend, no seasonality)
    spec = BSTSSpec(
        level=True,
        trend=True,  # Capture decline
        seasonal_periods=None,
        standardize_residual=True
    )
    
    logger.info("Decomposing with BSTS...")
    result = features(x, methods=['hvg', 'transition'], bsts=spec)
    
    # Display key results
    logger.info("\nKey Statistics:")
    logger.info(f"  Trend variance: {result.structural_stats.get('trend_variance', 0):.4f}")
    logger.info(f"  Residual std: {result.structural_stats.get('residual_std', 0):.4f}")
    
    hvg_stats = result.residual_network_stats.get('hvg', {})
    logger.info(f"  Residual HVG avg degree: {hvg_stats.get('avg_degree', 0):.2f}")
    
    # Flag wells with unstable decline (high residual complexity)
    residual_complexity = hvg_stats.get('avg_degree', 0)
    if residual_complexity > 3.5:
        logger.info("\n  Flag: High residual complexity - decline model may be unstable")
        logger.info("  Possible causes: regime shifts, operational changes, completion issues")
    else:
        logger.info("\n  Decline model stable - residual is mostly noise")


def example_windowed_analysis():
    """Windowed analysis for long series."""
    logger.info("\nWindowed Analysis Example: Long Series")
    
    # Generate long series (2 years of hourly data)
    np.random.seed(42)
    n_hours = 24 * 365 * 2  # 2 years
    
    t = np.arange(n_hours)
    level = 50 + 0.01 * t / 24
    daily = 10 * np.sin(2 * np.pi * t / 24)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    residual = np.random.randn(n_hours) * 2
    x = level + daily + weekly + residual
    
    logger.info(f"Generated {len(x)} hourly points ({len(x)/24/365:.1f} years)")
    
    # Use windowed analysis (auto-enabled for long series)
    spec = BSTSSpec(
        level=True,
        seasonal_periods=[24, 168],
        standardize_residual=True
    )
    
    logger.info("Decomposing with windowed analysis (window=10000)...")
    result = features(x, methods=['hvg'], bsts=spec, window=10000)
    
    # Windowed results are aggregated
    hvg_stats = result.residual_network_stats.get('hvg', {})
    logger.info("\nWindowed Residual Network Statistics:")
    logger.info(f"  Median avg degree: {hvg_stats.get('avg_degree_median', 0):.2f}")
    logger.info(f"  Q25-Q75 range: [{hvg_stats.get('avg_degree_q25', 0):.2f}, {hvg_stats.get('avg_degree_q75', 0):.2f}]")
    logger.info(f"  Number of windows: {hvg_stats.get('n_windows', 0)}")


def main():
    """Run all BSTS examples."""
    logger.info("BSTS Decomposition and Residual Topology Analysis")
    logger.info("=" * 60)
    
    example_smart_meter()
    example_well_production()
    example_windowed_analysis()
    
    logger.info("\n" + "=" * 60)
    logger.info("Examples complete!")
    logger.info("\nKey insights:")
    logger.info("- BSTS separates predictable structure from irregular dynamics")
    logger.info("- Residual topology flags series where structural model fails")
    logger.info("- Windowed analysis enables analysis of very long series")


if __name__ == "__main__":
    main()
