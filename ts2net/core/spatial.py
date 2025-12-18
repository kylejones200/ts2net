"""
Spatial analysis utilities for ts2net.

This module provides functions for spatial weights matrix generation.
"""

import numpy as np
from typing import Optional, Tuple


def radius_weights(coords: np.ndarray, radius: float) -> np.ndarray:
    """
    Generate spatial weights matrix based on radius threshold.
    
    Args:
        coords: Coordinate array of shape (n, d)
        radius: Distance threshold for connectivity
        
    Returns:
        Weights matrix of shape (n, n)
    """
    n = coords.shape[0]
    W = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= radius:
                W[i, j] = 1.0
                W[j, i] = 1.0
    
    return W


def knn_weights(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Generate spatial weights matrix based on k-nearest neighbors.
    
    Args:
        coords: Coordinate array of shape (n, d)
        k: Number of nearest neighbors
        
    Returns:
        Weights matrix of shape (n, n)
    """
    n = coords.shape[0]
    W = np.zeros((n, n))
    
    # Compute pairwise distances
    for i in range(n):
        dists = np.array([np.linalg.norm(coords[i] - coords[j]) for j in range(n)])
        dists[i] = np.inf  # Exclude self
        
        # Find k nearest neighbors
        neighbors = np.argsort(dists)[:k]
        
        for j in neighbors:
            W[i, j] = 1.0
    
    # Symmetrize
    W = np.maximum(W, W.T)
    
    return W

