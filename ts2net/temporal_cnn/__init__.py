"""
Temporal CNN for time series feature extraction.

Provides a simple 1D dilated CNN for fast, stable feature extraction
from time series windows.
"""

from .embeddings import temporal_cnn_embeddings

__all__ = ['temporal_cnn_embeddings']
