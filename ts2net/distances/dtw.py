"""
Dynamic Time Warping (DTW) distance implementation.

This module provides a pure Python implementation of the DTW algorithm
for cases where the optimized tslearn implementation is not available.
"""

import numpy as np


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series.

    Args:
        x, y: Input time series

    Returns:
        DTW distance between x and y
    """
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m]


import numpy as np
from tslearn.metrics import dtw


def compute_dtw_distance(ts1, ts2):
    return dtw(ts1, ts2)


if __name__ == "__main__":
    ts1 = np.array([1, 2, 3, 4, 5])
    ts2 = np.array([2, 3, 4, 5, 6])
    # Example usage - DTW distance computed
