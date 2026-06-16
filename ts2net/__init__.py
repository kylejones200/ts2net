"""
ts2net: Time Series to Networks

Clean API inspired by ts2vg, extended for multiple network methods.
"""

from .core.graph import Graph
from .api import HVG, NVG, RecurrenceNetwork, TransitionNetwork, build_network
from .core import graph_summary
from .exceptions import NotBuiltError, ValidationError, Ts2NetError
from .protocols import NetworkBuilder

__version__ = "0.8.0"

__all__ = [
    'Graph',
    'HVG',
    'NVG',
    'RecurrenceNetwork',
    'TransitionNetwork',
    'build_network',
    'graph_summary',
    'NetworkBuilder',
    'NotBuiltError',
    'ValidationError',
    'Ts2NetError',
]

# Optional Polars-based IO
try:
    from .io_polars import load_series_from_parquet_polars
    __all__.append('load_series_from_parquet_polars')
except ImportError:
    pass

# Visualization module
try:
    from .viz import (
        plot_series_with_events,
        plot_degree_profile,
        plot_degree_ccdf,
        plot_method_comparison,
        plot_window_feature_map,
        plot_hvg_small,
        plot_recurrence_matrix,
    )
    __all__.extend([
        'plot_series_with_events',
        'plot_degree_profile',
        'plot_degree_ccdf',
        'plot_method_comparison',
        'plot_window_feature_map',
        'plot_hvg_small',
        'plot_recurrence_matrix',
    ])
except ImportError:
    pass

# Columnar adapters
try:
    from .io_adapters import from_pandas, from_polars
    __all__.extend(['from_pandas', 'from_polars'])
except ImportError:
    pass

# Windowed graphs API
from .api_windows import build_windows
__all__.append('build_windows')

# Multiscale graphs API
from .multiscale import MultiscaleGraphs, coarse_grain
__all__.extend(['MultiscaleGraphs', 'coarse_grain'])

# Configuration and factory modules (pipeline extra for YAML file loading)
try:
    from .config import PipelineConfig
    from .factory import create_graph_builder, build_graph_from_config
    __all__.extend(['PipelineConfig', 'create_graph_builder', 'build_graph_from_config'])
except ImportError:
    pass

# BSTS decomposition and features (optional - requires statsmodels)
try:
    from .bsts import decompose, BSTSSpec, features
    __all__ = __all__ + ['decompose', 'BSTSSpec', 'features']
except ImportError:
    pass

# Temporal CNN embeddings (optional - requires torch)
try:
    from .temporal_cnn import temporal_cnn_embeddings
    __all__.append('temporal_cnn_embeddings')
except ImportError:
    pass

# Neural network-based network inference (optional - requires torch)
try:
    from .inference import NeuralNetworkInference, DynamicsModel, KuramotoModel, LinearDynamicsModel
    __all__.extend(['NeuralNetworkInference', 'DynamicsModel', 'KuramotoModel', 'LinearDynamicsModel'])
except ImportError:
    pass

# Causal inference (transfer entropy, Granger causality)
from .causal import (
    transfer_entropy,
    transfer_entropy_network,
    conditional_transfer_entropy,
    granger_causality,
    granger_causality_network,
    causal_strength,
    directionality_index,
    causal_network_metrics,
    time_lagged_causality_network,
    search_granger_lag,
    search_te_lag,
    run_causal_analysis,
    CausalWorkflowSpec,
    CausalAnalysisResult,
    pc_algorithm,
    pc_timeseries_network,
    PCResult,
    fci_algorithm,
    fci_timeseries_network,
    FCIResult,
    directed_visibility_analysis,
    visibility_irreversibility,
    VisibilityAsymmetryResult,
)
__all__.extend([
    'transfer_entropy',
    'transfer_entropy_network',
    'conditional_transfer_entropy',
    'granger_causality',
    'granger_causality_network',
    'causal_strength',
    'directionality_index',
    'causal_network_metrics',
    'time_lagged_causality_network',
    'search_granger_lag',
    'search_te_lag',
    'run_causal_analysis',
    'CausalWorkflowSpec',
    'CausalAnalysisResult',
    'pc_algorithm',
    'pc_timeseries_network',
    'PCResult',
    'fci_algorithm',
    'fci_timeseries_network',
    'FCIResult',
    'directed_visibility_analysis',
    'visibility_irreversibility',
    'VisibilityAsymmetryResult',
])

# Core graph expansion (v0.4)
from .graphs import (
    correlation_matrix,
    correlation_network,
    partial_correlation_matrix,
    partial_correlation_network,
    rolling_correlation_matrix,
    rolling_correlation_network,
    similarity_matrix,
    similarity_network,
    RollingGraphSequence,
    edge_persistence,
    graph_churn,
    edge_birth_death,
    MultiplexGraph,
    multiplex_graph,
    multiplex_visibility_graph,
    adaptive_recurrence_network,
    cross_recurrence_network,
    recurrence_matrix,
    recurrence_quantification,
    event_sequence_network,
    event_sync_network,
    sax_symbolize,
    entropy_max_symbolize,
    sax_transition_network,
)
__all__.extend([
    'correlation_matrix',
    'correlation_network',
    'partial_correlation_matrix',
    'partial_correlation_network',
    'rolling_correlation_matrix',
    'rolling_correlation_network',
    'similarity_matrix',
    'similarity_network',
    'RollingGraphSequence',
    'edge_persistence',
    'graph_churn',
    'edge_birth_death',
    'MultiplexGraph',
    'multiplex_graph',
    'multiplex_visibility_graph',
    'adaptive_recurrence_network',
    'cross_recurrence_network',
    'recurrence_matrix',
    'recurrence_quantification',
    'event_sequence_network',
    'event_sync_network',
    'sax_symbolize',
    'entropy_max_symbolize',
    'sax_transition_network',
])

# scikit-learn integration
from .sklearn import (
    NetworkFeatureExtractor,
    RollingNetworkFeatureExtractor,
    NetworkFeatureSelector,
    features_to_dataframe,
    compare_feature_sets,
    statistical_baseline_features,
)
__all__.extend([
    'NetworkFeatureExtractor',
    'RollingNetworkFeatureExtractor',
    'NetworkFeatureSelector',
    'features_to_dataframe',
    'compare_feature_sets',
    'statistical_baseline_features',
])

# Graph ML adapters (optional PyG/DGL)
try:
    from .ml import to_pyg_data, to_dgl_graph
    __all__.extend(['to_pyg_data', 'to_dgl_graph'])
except ImportError:
    pass

# Dynamic network analytics (0.8)
from .dynamic import (
    detect_regime_changes,
    window_anomaly_scores,
    edge_transition_anomalies,
    node_roles,
    node_role_evolution,
    community_labels,
    track_communities,
    run_dynamic_analysis,
    DynamicWorkflowSpec,
    DynamicAnalysisResult,
)
__all__.extend([
    'detect_regime_changes',
    'window_anomaly_scores',
    'edge_transition_anomalies',
    'node_roles',
    'node_role_evolution',
    'community_labels',
    'track_communities',
    'run_dynamic_analysis',
    'DynamicWorkflowSpec',
    'DynamicAnalysisResult',
])

# Scale and performance (0.6)
from .scale import (
    PerformanceContract,
    get_performance_contract,
    list_performance_contracts,
    iter_windows,
    iter_series_chunks,
    iter_parquet_value_chunks,
    build_windows_streaming,
    stream_chunk_stats,
    estimate_window_job_memory_mb,
    to_sparse_csr,
    edges_to_csr,
)
__all__.extend([
    'PerformanceContract',
    'get_performance_contract',
    'list_performance_contracts',
    'iter_windows',
    'iter_series_chunks',
    'iter_parquet_value_chunks',
    'build_windows_streaming',
    'stream_chunk_stats',
    'estimate_window_job_memory_mb',
    'to_sparse_csr',
    'edges_to_csr',
])
