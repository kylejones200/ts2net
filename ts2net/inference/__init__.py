"""
Neural network-based network inference from time series.

Provides functionality to infer network adjacency matrices from observed
time series data using neural networks with uncertainty quantification.
"""

try:
    from .neural_inference import (
        NeuralNetworkInference,
        DynamicsModel,
        KuramotoModel,
        LinearDynamicsModel,
        AdjacencyNetwork,
    )
    __all__ = [
        "NeuralNetworkInference",
        "DynamicsModel",
        "KuramotoModel",
        "LinearDynamicsModel",
        "AdjacencyNetwork",
    ]
except ImportError:
    __all__ = []

